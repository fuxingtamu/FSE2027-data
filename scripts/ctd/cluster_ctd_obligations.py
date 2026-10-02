# -*- coding: utf-8 -*-
"""Cluster per-case CTD judgments into distinct obligation-level results.

The script reads enriched SZZ cases, asks an LLM to group related cases,
validates that every candidate is assigned exactly once, and writes resumable
checkpoints plus the final obligation dataset.
"""
import json, os, re, sys, time, argparse, random, threading
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import ijson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
OUTPUT_DIR = SRC_DIR / "output"

MERGED_FULL = OUTPUT_DIR / "final" / "_td_classifier_blind_gpt-5.6-luna.merged_full.json"
CKPT_FILE = OUTPUT_DIR / "_intermediate" / "_td_obligation_ckpt.jsonl"
OUT_JSON = OUTPUT_DIR / "final" / "_td_obligations_gpt-5.6-luna.json"

# Implementation note.
for p in [Path(".env"), PROJECT_ROOT / ".env"]:
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    raw = line
                    if raw.startswith("export "):
                        raw = raw[7:]
                    k, v = raw.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        break

# Prefer the explicitly requested V endpoint when it is configured.  Keep the
# legacy variables as a fallback for older experiments.
LLM_BASE_URL = os.environ.get("V_LLM_BASE_URL", os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.huiyan-ai.cn/v1")))
LLM_MODEL = os.environ.get("V_LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-5.6-luna"))

CATS = ['TD-1', 'TD-2', 'TD-3', 'TD-4', 'TD-5', 'TD-6', 'TD-7',
        'TD-8', 'TD-9', 'TD-10', 'TD-11']

# ═══════════════════════════════════════════════════════════════════════════
# Implementation note.
# ═══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an annotator studying technical debt (TD) in AI-generated code.

An SZZ pipeline has identified N debt candidate sites in an AI-authored PR. Each site is a region of AI-introduced code later touched by a fix commit and has already received an independent judgment for TD status, category, subtype, nature, and reasoning. Your task is not to re-decide whether each site is TD. Cluster the candidate sites into distinct technical-debt obligations.

What is one obligation?
An obligation is one distinct debt introduced by AI-authored code and later repaid. The unit is the debt itself, not a file, fix commit, or PR. One obligation may span several files and be repaid through several commits.

Merge candidate sites into one obligation when any of these apply:
1. Their per-site reasoning describes the same underlying defect or omission (for example, multiple files lack a null check for the same API contract).
2. One omission propagates across multiple files or locations (for example, one missing guard appears in five files).
3. Several fix commits are stages in repaying the same debt rather than unrelated repairs.
4. One AI-authored code region was split across multiple repair changes.

Split candidate sites into separate obligations when any of these apply:
1. They have different root causes or describe different defects.
2. Their TD categories differ; different categories almost always indicate separate obligations.
3. They share a category or subtype but are distinct instances in separate functions or modules that can be repaired independently. The key question is whether they are the same defect, not whether they share a label.
4. Fixing one site would not affect the other.

Decision guidance:
- Read each site's per-case reasoning first. The same defect supports merging; distinct defects support splitting.
- Different subtypes (for example, 3.1 and 3.2) usually indicate separate obligations. Merge only if the reasoning clearly shows two aspects of the same defect, and explain why.
- If a site's original judgment is clearly a false positive, place it in `flagged_not_td` and explain why.
- Do not merge every site into one obligation or split excessively. K should represent the actual number of distinct obligations.

Output only one JSON object, without Markdown fences:
{
  "obligations": [
    {
      "obligation_id": "O1",
      "description": "One-sentence description of the debt.",
      "root_cause": "What the AI implementation did wrong.",
      "member_indices": ["C1", "C3", "C7"],
      "category": "TD-X",
      "subtype": "X.Y subtype name",
      "nature": "DEFECT_REPAIR" | "QUALITY_CLEANUP",
      "affected_files": ["..."],
      "repayment_commits": ["first 8 characters of commit SHA..."],
      "confidence": 0.0
    }
  ],
  "flagged_not_td": ["C2"],
  "notes": "Optional explanation of merges or splits, especially cross-subtype or multi-file merges."
}

Hard constraints:
1. Every candidate index (C1..CN) must appear exactly once, either in one obligation's member_indices or in flagged_not_td.
2. No member index may occur in more than one obligation.
3. If a candidate cannot be classified confidently, put it in flagged_not_td rather than forcing it into an obligation.
4. Derive affected_files and repayment_commits from the corresponding candidate sites.
"""


def build_user_prompt(pr_key: str, pr_info: dict, candidates: list) -> str:
    """Format PR metadata and candidate evidence for obligation clustering."""
    parts = [f"## PR OVERVIEW: {pr_key}"]
    if pr_info.get("title"):
        parts.append(f"Title: {pr_info['title']}")
    parts.append(
        f"Agent: {pr_info.get('agent', '?')} | Merged: {pr_info.get('merged', '?')} "
        f"| Repo: {pr_info.get('repo', '?')}"
    )
    body = pr_info.get("body", "") or ""
    if body:
        body = body[:1200] + ("\n... [truncated]" if len(body) > 1200 else "")
        parts.append(f"PR Description:\n{body}")

    parts.append("")
    parts.append(
        f"## {len(candidates)} Candidate Sites\n"
        "Each site links an AI-authored code region to a later fix commit:"
    )
    parts.append("")
    for candidate in candidates:
        parts.append(f"[{candidate['idx']}] file: {candidate['file']}")
        parts.append(
            f"     fix commit: {candidate['commit_short']} | {candidate['commit_msg']} "
            f"| {candidate['commit_date']} | latency {candidate['latency']}d "
            f"| verdict={candidate['verdict']}"
        )
        parts.append(
            f"     per-case: {candidate['cat']} / {candidate['subtype']} / "
            f"{candidate['nature']} / confidence {candidate['conf']}"
        )
        parts.append(f"     reasoning: {candidate['reasoning']}")
        parts.append("")

    parts.append(
        f"Cluster these {len(candidates)} candidate sites into distinct obligations. "
        "Follow the system instructions and return exactly one JSON object."
    )
    return "\n".join(parts)

def call_llm(system: str, user: str, model: str = None, max_retries: int = 6,
             max_tokens: int = 8000, debug: bool = False) -> dict:
    """Call the configured API and return a parsed JSON object or error record."""
    import urllib.request
    model = model or LLM_MODEL
    api_keys = []
    main_key = os.environ.get("V_OPENAI_API_KEY", "")
    if main_key:
        api_keys.append(main_key)
    for i in range(1, 11):
        k = os.environ.get(f"V_OPENAI_API_KEY{i}", "")
        if k and k not in api_keys:
            api_keys.append(k)
    # Fallback only when no V keys are configured.
    if not api_keys:
        main_key = os.environ.get("OPENAI_API_KEY", "")
        if main_key:
            api_keys.append(main_key)
        for i in range(1, 11):
            k = os.environ.get(f"OPENAI_API_KEY{i}", "")
            if k and k not in api_keys:
                api_keys.append(k)
    if not api_keys:
        raise RuntimeError("No OpenAI API key found")

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    last_error = None
    for attempt in range(max_retries):
        api_key = random.choice(api_keys)
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode('utf-8'),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))
            resp = urllib.request.urlopen(req, timeout=timeout_seconds)
            data = json.loads(resp.read())
            content = (data["choices"][0]["message"]["content"] or "").strip()
            finish = data.get("choices", [{}])[0].get("finish_reason", "")
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            # Accept the first complete JSON object if a model appends a
            # second object or explanatory text after the answer.
            decoder = json.JSONDecoder()
            for start, char in enumerate(content):
                if char != '{':
                    continue
                try:
                    parsed, _ = decoder.raw_decode(content[start:])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
            # Implementation note.
            if debug:
                print(f"    [call_llm] Response had no JSON object; finish={finish}; tail={content[-200:]!r}")
            raise ValueError(f"no JSON object in response (finish={finish})")
        except Exception as e:
            last_error = e
            if debug and attempt > 0:
                print(f"    [call_llm] Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2.0 ** attempt, 90.0))
    return {"error": str(last_error)}


# ═══════════════════════════════════════════════════════════════════════════
# Implementation note.
# ═══════════════════════════════════════════════════════════════════════════
def load_td_cases():
    """Load TD-positive cases and group them by repository and pull request."""
    pr_cases = defaultdict(list)
    with open(MERGED_FULL, 'rb') as fh:
        for it in ijson.items(fh, 'item', use_float=True):
            if it.get('_llm_is_td') is not True:
                continue
            repo = it.get('repo_name') or '?'
            pr = it.get('pr_number') or '?'
            pr_cases[(repo, pr)].append(it)
    return dict(pr_cases)


def candidate_view(c, idx):
    """Build the compact evidence record used in the clustering prompt."""
    fd = c.get('fix_diff', {})
    if isinstance(fd, dict):
        additions = fd.get('additions', 0)
        deletions = fd.get('deletions', 0)
    else:
        additions = deletions = 0
    commit_msg = (c.get('commit_message') or '?').replace('\n', ' ').strip()[:120]
    reasoning = (c.get('_llm_reasoning') or '').strip()
    if len(reasoning) > 280:
        reasoning = reasoning[:280] + "..."
    return {
        'idx': f"C{idx}",
        'case_id': c.get('case_id') or '?',
        'file': c.get('filepath') or '?',
        'commit_short': (c.get('commit_sha') or '?')[:8],
        'commit_sha': c.get('commit_sha') or '?',
        'commit_msg': commit_msg,
        'commit_date': c.get('commit_date') or '?',
        'latency': c.get('fix_latency_days') or '?',
        'verdict': c.get('verdict_level') or '?',
        'cat': c.get('_llm_td_category') or '?',
        'subtype': c.get('_llm_td_subtype') or '?',
        'nature': c.get('_llm_nature') or '?',
        'conf': c.get('_llm_confidence') or 0,
        'reasoning': reasoning,
        'add': additions, 'del': deletions,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Implementation note.
# ═══════════════════════════════════════════════════════════════════════════
def validate_clustering(indices: list, parsed: dict):
    """Check that every candidate is assigned exactly once."""
    errors = []
    seen = []
    for o in parsed.get('obligations', []):
        for idx in o.get('member_indices', []):
            if idx in seen:
                errors.append(f"Candidate {idx} appears more than once")
            seen.append(idx)
    for idx in parsed.get('flagged_not_td', []):
        if idx in seen:
            errors.append(f"Candidate {idx} appears in both an obligation and flagged_not_td")
        seen.append(idx)
    missing = [i for i in indices if i not in seen]
    if missing:
        errors.append(f"Candidates missing from the result: {missing}")
    extra = [i for i in seen if i not in indices]
    if extra:
        errors.append(f"Unexpected candidate indices: {extra}")
    return (len(errors) == 0), errors


# ═══════════════════════════════════════════════════════════════════════════
# Implementation note.
# ═══════════════════════════════════════════════════════════════════════════
def cluster_pr(pr_key: str, cases: list, model: str = None, debug: bool = False):
    """Cluster one pull request and return its validated obligation records."""
    repo, pr = pr_key
    pr_key_slug = f"{repo.split('/')[-1]}#{pr}"
    c0 = cases[0]
    pr_info = {
        'title': (c0.get('pr_title') or '').strip()[:200] or 'Unknown',
        'agent': c0.get('agent') or '?',
        'merged': c0.get('pr_merged_at') or '?',
        'repo': repo,
        'body': (c0.get('pr_body') or '')[:2000],
    }

    if len(cases) == 1:
        c = cases[0]
        reasoning = (c.get('_llm_reasoning') or '').strip() or 'single candidate'
        return [{
            'obligation_key': pr_key_slug, 'obligation_index': 1, 'k': 1,
            'source': 'single',
            'description': reasoning[:200],
            'root_cause': reasoning,
            'member_case_ids': [c.get('case_id') or '?'],
            'member_indices': ['C1'],
            'category': c.get('_llm_td_category') or '?',
            'subtype': c.get('_llm_td_subtype') or '?',
            'nature': c.get('_llm_nature') or '?',
            'confidence': c.get('_llm_confidence') or 0,
            'affected_files': [c.get('filepath') or '?'],
            'repayment_commits': [(c.get('commit_sha') or '?')[:8]],
            'repo': repo, 'pr': pr, 'pr_title': pr_info['title'], 'agent': pr_info['agent'],
            'merged_at': pr_info['merged'],
        }]

    # Implementation note.
    candidates = [candidate_view(c, i + 1) for i, c in enumerate(cases)]
    indices = [c['idx'] for c in candidates]
    user = build_user_prompt(pr_key_slug, pr_info, candidates)
    # Implementation note.
    max_tokens = min(20000, max(8000, len(candidates) * 250))

    parsed = None
    last_errs = []
    for attempt in range(4):
        resp = call_llm(SYSTEM_PROMPT, user, model=model, max_tokens=max_tokens, debug=debug)
        if resp is None or 'error' in resp:
            last_errs = [str(resp)]
            if attempt < 3:
                user += "\n\nThe previous response was invalid. Return one JSON object matching the required schema."
            continue
        if 'obligations' not in resp:
            last_errs = ["The model returned an invalid obligation assignment."]
            if attempt < 3:
                user += "\n\nReturn one valid JSON object with every candidate assigned exactly once."
            continue
        ok, errs = validate_clustering(indices, resp)
        if ok:
            parsed = resp
            break
        last_errs = errs
        if attempt < 3:
                user += "\n\nThe previous response was invalid. Return one JSON object matching the required schema."

    if parsed is None:
        # Implementation note.
        obls = []
        for c in candidates:
            obls.append({
                'obligation_key': pr_key_slug, 'obligation_index': 0, 'k': len(candidates),
                'source': 'clustering_error',
                'description': c['reasoning'][:200],
                'root_cause': c['reasoning'],
                'member_case_ids': [c['case_id']],
                'member_indices': [c['idx']],
                'category': c['cat'], 'subtype': c['subtype'], 'nature': c['nature'],
                'confidence': c['conf'],
                'affected_files': [c['file']],
                'repayment_commits': [c['commit_short']],
                'repo': repo, 'pr': pr, 'pr_title': pr_info['title'], 'agent': pr_info['agent'],
                'merged_at': pr_info['merged'],
                'clustering_error': str(last_errs),
            })
        return obls

    # Implementation note.
    cid_by_idx = {c['idx']: c for c in candidates}
    obls = []
    for i, o in enumerate(parsed['obligations'], 1):
        members = [cid_by_idx[idx] for idx in o.get('member_indices', []) if idx in cid_by_idx]
        obls.append({
            'obligation_key': pr_key_slug, 'obligation_index': i, 'k': len(parsed['obligations']),
            'source': 'llm_cluster',
            'description': o.get('description', ''),
            'root_cause': o.get('root_cause', ''),
            'member_case_ids': [m['case_id'] for m in members],
            'member_indices': o.get('member_indices', []),
            'category': o.get('category') or '?',
            'subtype': o.get('subtype') or '',
            'nature': o.get('nature') or '?',
            'confidence': o.get('confidence') or 0,
            'affected_files': o.get('affected_files', [m['file'] for m in members]),
            'repayment_commits': o.get('repayment_commits', []),
            'repo': repo, 'pr': pr, 'pr_title': pr_info['title'], 'agent': pr_info['agent'],
            'merged_at': pr_info['merged'],
            'notes': parsed.get('notes', ''),
        })
    for idx in parsed.get('flagged_not_td', []):
        if idx in cid_by_idx:
            obls.append({
                'obligation_key': pr_key_slug, 'obligation_index': 0, 'k': len(parsed['obligations']),
                'source': 'flagged_not_td',
                'description': f"Case rationale: {cid_by_idx[idx]['reasoning'][:150]}",
                'root_cause': cid_by_idx[idx]['reasoning'],
                'member_case_ids': [cid_by_idx[idx]['case_id']],
                'member_indices': [idx],
                'category': cid_by_idx[idx]['cat'], 'subtype': cid_by_idx[idx]['subtype'],
                'nature': cid_by_idx[idx]['nature'],
                'confidence': cid_by_idx[idx]['conf'],
                'affected_files': [cid_by_idx[idx]['file']],
                'repayment_commits': [cid_by_idx[idx]['commit_short']],
                'repo': repo, 'pr': pr, 'pr_title': pr_info['title'], 'agent': pr_info['agent'],
                'merged_at': pr_info['merged'],
                'notes': 'flagged by obligation clustering as not actually TD',
            })
    return obls


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="TD Obligation-Level Clustering")
    parser.add_argument("--model", default=None, help=f"LLM model name (default: {LLM_MODEL})")
    parser.add_argument("--limit-pr", type=int, default=0)
    parser.add_argument("--pr", action="append", default=[], metavar="SLUG#N",
                        help="Filter by repository/pull-request key, for example eliza#16062")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the LLM")
    parser.add_argument("--debug", action="store_true", help="Enable verbose LLM diagnostics")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_FILE))
    parser.add_argument("--out", type=str, default=str(OUT_JSON))
    args = parser.parse_args()

    print("Loading CTD cases from merged_full JSON...")
    pr_cases = load_td_cases()
    total_cases = sum(len(v) for v in pr_cases.values())
    print(f"  Pull requests with TD cases: {len(pr_cases)}; candidate cases: {total_cases}")

    single = {k for k, v in pr_cases.items() if len(v) == 1}
    multi = {k for k, v in pr_cases.items() if len(v) > 1}
    print(f"  Single-case PRs: {len(single)}; multi-case PRs: {len(multi)}")

    # Implementation note.
    targets = []
    if args.pr:
        wanted = set()
        for spec in args.pr:
            slug, _, num = spec.rpartition('#')
            wanted.add((slug, num))
        for k in pr_cases:
            slug = k[0].split('/')[-1]
            if (slug, str(k[1])) in wanted:
                targets.append(k)
        print(f"  Target pull requests: {len(targets)}")
    else:
        targets = sorted(pr_cases.keys())
        if args.limit_pr > 0:
            targets = targets[:args.limit_pr]
    # Implementation note.
        print(f"  Target pull requests: {len(targets)}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN: showing prompts for up to two pull requests; no LLM calls will be made")
        print("=" * 70)
        shown = 0
        for k in targets:
            if len(pr_cases[k]) < 2:
                continue
            slug = f"{k[0].split('/')[-1]}#{k[1]}"
            cands = [candidate_view(c, i + 1) for i, c in enumerate(pr_cases[k])]
            user = build_user_prompt(slug, {'title': (pr_cases[k][0].get('pr_title') or '?'),
                                            'agent': pr_cases[k][0].get('agent'),
                                            'merged': pr_cases[k][0].get('pr_merged_at'),
                                            'repo': k[0], 'body': pr_cases[k][0].get('pr_body')}, cands)
            tok = len(user) // 2  # Rough token estimate
            print(f"\n--- {slug} | {len(cands)} candidates | ~{tok} tokens ---")
            print(user[:1500])
            if tok > 60000:
                print("  WARNING: prompt estimated above 60K tokens; skipping")
            shown += 1
            if shown >= 2:
                break
        print("\nDRY-RUN done.")
        return

    # Implementation note.
    done = set()
    obligations = []
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                obligations.append(rec)
                done.add(rec['obligation_key'])
            except Exception:
                continue
        print(f"  [checkpoint] {len(done)} pull requests already completed")

    todo = [(k, pr_cases[k]) for k in targets if f"{k[0].split('/')[-1]}#{k[1]}" not in done]
    print(f"  Pull requests remaining: {len(todo)}")

    # Implementation note.
    results = []
    rlock = threading.Lock()
    clk = threading.Lock()
    completed = [0]
    ckpt_handle = open(ckpt_path, 'a', encoding='utf-8') if ckpt_path else None

    def cluster_one(pr_key, cases):
        slug = f"{pr_key[0].split('/')[-1]}#{pr_key[1]}"
        obls = cluster_pr(pr_key, cases, model=args.model, debug=args.debug)
        with rlock:
            results.extend(obls)
            if ckpt_handle:
                for o in obls:
                    ckpt_handle.write(json.dumps(o, ensure_ascii=False) + '\n')
                ckpt_handle.flush()
        with clk:
            completed[0] += 1
            k = obls[0]['k'] if obls else 0
            src = obls[0]['source'] if obls else '?'
            print(f"  [{completed[0]}/{len(todo)}] {slug} -> K={k} ({src})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(cluster_one, k, v): (k, v) for k, v in todo}
        for f in as_completed(futures):
            f.result()

    if ckpt_handle:
        ckpt_handle.close()

    # Implementation note.
    total_obls = obligations + results
    print(f"\nTotal obligations: {len(total_obls)} across {len(pr_cases)} PRs")
    cat = Counter(o['category'] for o in total_obls if o['category'] in CATS)
    print("By category:")
    for c in CATS:
        if cat[c]:
            print(f"  {c}: {cat[c]} ({cat[c]/sum(cat.values())*100:.1f}%)")
    src = Counter(o['source'] for o in total_obls)
    print("By source:", dict(src))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({'total_obligations': len(total_obls), 'total_prs': len(pr_cases),
               'category_dist': dict(cat), 'source_dist': dict(src),
               'obligations': total_obls},
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"written -> {args.out}")


if __name__ == '__main__':
    main()

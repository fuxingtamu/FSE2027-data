"""Semantic classification of PR purpose for the frozen RQ1 AI-PR set."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/phase2_szz"))
import td_classifier_obligation as api  # noqa: E402

ROSTER = ROOT / "src/output/human_control/full_roster_matching/ai_matching_cohort.json"
META = ROOT / "src/output/repo_selection/pr_meta_cache.json"
OUTDIR = ROOT / "src/output/human_control/purpose_classification_rq1"
CONTROL_MANIFEST = OUTDIR / "control_purpose_candidates_2438.json"
CONTROL_META = OUTDIR / "control_purpose_pr_metadata.json"
PURPOSES = {
    "bug-fix", "feature", "refactor", "test", "documentation",
    "configuration-dependency", "maintenance", "mixed-purpose", "other", "unknown",
}

SYSTEM = """You classify the primary purpose of a merged pull request for an empirical
software-engineering study. Use only the PR title, description, changed-file list,
and origin diff supplied by the user. Do not infer AI authorship or use any later
repair/technical-debt information. Judge the intent of the whole PR from the evidence,
not from keywords or filenames alone. Assign exactly one label:

- bug-fix: correct an existing defect, failure, regression, or incorrect behavior.
- feature: add a user/developer capability or supported behavior.
- refactor: reorganize or simplify existing behavior without an intended behavior change.
- test: tests or test infrastructure are the main deliverable, with no primary product fix or feature.
- documentation: documentation or examples are the main deliverable.
- configuration-dependency: build, CI, deployment, configuration, or dependency changes are the main goal.
- maintenance: routine upgrades, cleanup, formatting/style, release, or repository upkeep; use for performance tuning only when that is the stated main goal and not a bug fix or new capability.
- mixed-purpose: two or more distinct purposes are equally central and no single primary purpose is supported by the evidence. Use sparingly.
- other: a clear primary purpose not covered above.
- unknown: the supplied evidence is insufficient to determine the purpose.

Boundary rules: tests supporting a bug fix remain bug-fix; tests supporting a new feature remain
feature. A test-only PR is test. A documentation-only PR is documentation. A change called a
"refactor" that actually changes behavior should be classified by the behavior change. Prefer
the purpose supported by the diff when title/body wording conflicts with the actual change.

Return one JSON object only:
{"purpose":"<label>","confidence":0.0,"reason":"brief explanation","evidence":["short evidence from title/body/diff"]}
"""


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[middle omitted]...\n" + text[-half:]


def load_records(group: str = "ai") -> list[dict]:
    if group == "control":
        roster = json.loads(CONTROL_MANIFEST.read_text(encoding="utf-8"))
        cache = json.loads(CONTROL_META.read_text(encoding="utf-8"))
        records = []
        for r in roster:
            key = f"{r['repo']}#{r['number']}"
            m = cache.get(key)
            if not m:
                raise ValueError(f"Missing control PR metadata: {key}")
            files = m.get("files") or []
            patches = [f"### {f.get('filename', '?')}\n{f.get('patch', '')}"
                       for f in files if f.get("patch")]
            title = m.get("title") or r.get("title") or ""
            records.append({
                "key": key, "repo": r["repo"], "number": str(r["number"]),
                "title": title, "body": m.get("body") or "", "files": files,
                "patch": "\n\n".join(patches),
                # Only supports stratified smoke sampling; never sent to the model.
                "old_heuristic": r.get("pr_type", "other"),
            })
        if len(records) != 2438 or len({r["key"] for r in records}) != 2438:
            raise ValueError(f"Unexpected control-purpose roster: {len(records)} records")
        return records

    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    # This is the RQ1 scope used by the paper: drop the documentation-only repo.
    roster = [r for r in roster if r.get("repo") != "dotnet/docs"]
    if len(roster) != 2717 or len({r.get("repo") for r in roster}) != 20:
        raise ValueError(f"Unexpected frozen RQ1 roster: {len(roster)} PRs")
    cache = json.loads(META.read_text(encoding="utf-8"))
    records = []
    for r in roster:
        key = f"{r['repo']}#{r['number']}"
        m = cache.get(key)
        if not m:
            raise ValueError(f"Missing cached PR metadata: {key}")
        files = m.get("files") or []
        patches = [f"### {f.get('filename', '?')}\n{f.get('patch', '')}"
                   for f in files if f.get("patch")]
        title = m.get("title") or r.get("title") or ""
        body = m.get("body") or ""
        heuristic_text = f"{title} {body}".lower()
        if any(x in heuristic_text for x in ("fix", "bug", "hotfix", "regression", "error")):
            heuristic = "bug-fix"
        elif any(x in heuristic_text for x in ("refactor", "restructure", "cleanup", "clean-up", "reorganize", "migrate")):
            heuristic = "refactor"
        elif any(x in heuristic_text for x in ("feat", "feature", "implement", "support", "add ", "introduce")):
            heuristic = "feature"
        else:
            heuristic = "other"
        records.append({
            "key": key,
            "repo": r["repo"],
            "number": str(r["number"]),
            "title": title,
            "body": body,
            "files": files,
            "patch": "\n\n".join(patches),
            # Used only to stratify the smoke sample; never sent to the model.
            "old_heuristic": heuristic,
        })
    return records


def prompt_for(r: dict) -> str:
    file_rows = []
    for f in r["files"]:
        file_rows.append(
            f"- {f.get('filename', '?')} (+{f.get('additions', 0)} "
            f"-{f.get('deletions', 0)})"
        )
    return "\n".join([
        f"PR title: {r['title']}",
        "PR description:", clip(r["body"], 5000) or "(not available)",
        "Changed files:", "\n".join(file_rows) or "(not available)",
        "Origin PR diff:", clip(r["patch"], 16000) or "(not available)",
    ])


def valid(j: object) -> bool:
    return (isinstance(j, dict) and j.get("purpose") in PURPOSES
            and isinstance(j.get("confidence"), (int, float))
            and isinstance(j.get("reason"), str)
            and isinstance(j.get("evidence"), list))


def smoke_sample(records: list[dict], count: int) -> list[dict]:
    rng = random.Random(20260923)
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        strata[r["old_heuristic"]].append(r)
    for group in strata.values():
        rng.shuffle(group)
    picked = []
    labels = sorted(strata)
    while len(picked) < count and any(strata.values()):
        for label in labels:
            if strata[label] and len(picked) < count:
                picked.append(strata[label].pop())
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="classify a small stratified pilot")
    ap.add_argument("--smoke-count", type=int, default=24)
    ap.add_argument("--group", choices=("ai", "control"), default="ai")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=int, default=35,
                    help="per-request timeout in seconds")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    os.environ["LLM_TIMEOUT_SECONDS"] = str(args.timeout)
    # Use Huiyan's separately configured endpoint and key family. The shared
    # API helper normally gives the V_* endpoint precedence, so override it
    # explicitly and remove those keys from this process environment.
    api.LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
    api.LLM_MODEL = os.environ.get("LLM_MODEL", api.LLM_MODEL)
    for name in ("V_OPENAI_API_KEY", *(f"V_OPENAI_API_KEY{i}" for i in range(1, 11))):
        os.environ.pop(name, None)
    host = urlparse(api.LLM_BASE_URL).hostname
    if host != "api.huiyan-ai.cn":
        raise RuntimeError(f"Huiyan endpoint is not configured (host={host!r})")
    if not any(os.environ.get(f"OPENAI_API_KEY{i}") for i in range(1, 11)) and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Huiyan API credentials are not configured")
    model = args.model or api.LLM_MODEL
    records = load_records(args.group)
    if args.smoke:
        records = smoke_sample(records, args.smoke_count)
    default_name = (
        f"purpose_smoke_{args.group}_huiyan.jsonl" if args.smoke
        else ("purpose_labels_huiyan.jsonl" if args.group == "ai"
              else "purpose_labels_control_huiyan.jsonl")
    )
    out = Path(args.output) if args.output else OUTDIR / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    if row.get("valid"):
                        completed[row["key"]] = row
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in records if r["key"] not in completed]
    print(f"group={args.group} roster={len(records)} already_valid={len(completed)} pending={len(todo)}", flush=True)

    def classify(r: dict) -> dict:
        p = prompt_for(r)
        judgment = api.call_llm(SYSTEM, p, model=model, max_retries=2, max_tokens=1200)
        return {
            "key": r["key"], "repo": r["repo"], "pr_number": r["number"],
            "model": model, "endpoint": host, "prompt_chars": len(p),
            "judgment": judgment, "valid": valid(judgment),
        }

    with out.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(classify, r) for r in todo]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(futures):
                    print(f"progress={i}/{len(futures)} valid={row['valid']} key={row['key']}", flush=True)
    all_rows = {}
    with out.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("valid"):
                    all_rows[row["key"]] = row
    counts = Counter(r["judgment"].get("purpose", "invalid") for r in all_rows.values())
    print(f"group={args.group} saved={out} valid_total={sum(counts.values())} purposes={dict(counts)}", flush=True)


if __name__ == "__main__":
    main()

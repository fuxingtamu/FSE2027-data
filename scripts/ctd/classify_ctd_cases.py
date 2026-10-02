"""
TD Classifier (Simplified) — Pure TD Judgment, No Filters
==========================================================
Classify later fixes against AI-authored pull-request code and label consequential technical debt.
Classify later fixes against AI-authored pull-request code and label consequential technical debt.

Usage:
  python anonymous_release/scripts/ctd/classify_ctd_cases.py --dry-run
  python anonymous_release/scripts/ctd/classify_ctd_cases.py --limit-pr 10
  python anonymous_release/scripts/ctd/classify_ctd_cases.py
"""

import json, os, re, sys, time, argparse
from pathlib import Path
from collections import defaultdict, Counter
from math import isfinite
from typing import Dict, Any, List, Optional

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
OUTPUT_DIR = SRC_DIR / "output"

# ── API keys ──
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

ALL_REPOS = ["crewAI", "firecrawl", "gluesql", "lingo.dev", "onlook", "pdfme", "vikunja", "vscode"]

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an annotator studying technical debt (TD) in AI-generated code.

Given an AI-authored pull request (PR) that was merged after review and a later fix commit that modifies the same file, determine whether the fix repays technical debt introduced by the AI PR. If so, classify the debt.

Core question: Did the AI-authored code withstand the repository's subsequent evolution?

A later edit touching AI-authored lines is not, by itself, technical debt. Apply the Semantic Connection Test: why did the fix modify the AI-authored code? External changes such as new requirements, architectural refactoring, major-release cleanup, or dependency changes are triggers, not automatic exemptions. Determine how the external change affected the AI code.

- If the AI code itself had a defect (incorrect logic, weak typing, missing edge-case handling, or repository-style violations), label TD.
- If an external evolution merely scans or extends the code while preserving its structure, label NOT TD.
- If supporting the evolution requires dismantling, rewriting, or restructuring the AI code, this is strong evidence of structural debt (TD-5).
- If the AI code is untouched, label NOT TD.

When to label TD:
1. The fix corrects a bug rooted in the AI-authored code (for example, an incorrect loop condition): TD-1.
2. The fix adds error handling, boundary checks, or parameter validation missing from the AI code: TD-3.
3. The fix corrects a type problem, such as replacing an unsafe `any` type: TD-4.
4. The fix brings AI-authored code into compliance with established repository formatting or lint rules: TD-11.
5. The fix restructures highly coupled or duplicated AI-authored code: TD-5.

Evaluate external-evolution cases from the diff, not just commit-title keywords:
1. Release cleanup: incidental removal of unrelated tests or dead code is NOT TD; removal of the AI implementation because of its quality can be TD.
2. Repository-wide refactoring: incidental inclusion of nearby AI code is NOT TD; refactoring driven by excessive coupling, duplication, or poor extensibility in the AI code can be TD-5.
3. New feature: adding fields or branches while preserving the original AI structure is NOT TD; restructuring that code to support the feature can be TD-5.
4. Toolchain or dependency adaptation: changes required across all files are NOT TD; a bad coupling introduced by the AI code that forces its logic to be rewritten can be TD.
5. Feature removal: a product decision unrelated to code quality is NOT TD; removal caused by an unmaintainable or poor-quality AI implementation can be TD.

Boundary cases that can be TD:
- The AI copied an existing bad pattern (often TD-5.3 or TD-2.3).
- The implementation was unavoidable at merge time but later needed restructuring (often TD-5.2 or TD-5.3).
- The fix replaces AI-authored code with a better implementation in the same location.

Use UNKNOWN only when the evidence is genuinely contradictory or insufficient. For example, a whole-file deletion may reflect a rename or directory refactor rather than a quality problem, especially when the commit message and diff conflict. UNKNOWN means the evidence does not support a decision; it is not a way to avoid a difficult decision.

Before returning JSON, the reasoning must cover these three steps:
1. State the factual overlap: identify which fix-diff lines modify or delete which AI-PR patch lines, or state that no AI-authored code lines overlap and explain why.
2. Explain why the fix changed the AI code: an AI-code defect (TD), incidental contact during external evolution (NOT TD), or unresolved evidence (UNKNOWN).
3. If TD, explain the taxonomy assignment.

Evidence guidance:
- Use the commit message and PR description as important clues about why the AI code changed.
- Do not substitute SZZ trace labels for your own semantic judgment. SZZ indicates whether the fix overlaps AI-authored code, not why.
- PURE_ADDITIVE_NEAR is easy to misclassify. Inspect each nearby addition. Defensive wrappers such as try/catch, null checks, boundary checks, or input validation may repair a defect (often TD-3), but ordinary feature expansion is NOT TD. Structural changes for a new requirement must be judged using the external-evolution rules. Lines added near AI code are marked `[ADDED NEAR AI CODE]`.
- SZZ_PASSED means the fix deleted AI-authored lines; it does not prove TD. The deletion may correct poor AI code or result from external evolution.

Technical-debt taxonomy:
- TD-1 Algorithmic Debt: local algorithm or computation problems. Subtypes: 1.1 Incorrect Logic; 1.2 Inefficient Approach; 1.3 Data Structure Misuse.
- TD-2 Contractual Debt: interface or contract problems. Subtypes: 2.1 Signature Mismatch; 2.2 Abstraction Leak; 2.3 Pattern Inconsistency.
- TD-3 Defensive Debt: missing or fragile handling of exceptional paths and boundaries. Subtypes: 3.1 Error Handling Gap; 3.2 Edge Case Omission; 3.3 State Fragility (concurrency, ordering, or lifecycle).
- TD-4 Type Debt: unsafe or inappropriate type-system use. Subtypes: 4.1 Type Looseness; 4.2 Null Unsafety; 4.3 Unsafe Coercion.
- TD-5 Structural Debt: module or component design problems such as excessive coupling, low cohesion, or duplication. Subtypes: 5.1 Coupling Excess; 5.2 Cohesion Deficit; 5.3 Duplication.
- TD-6 Architectural Debt: system-level design problems such as dependency-rule violations, architectural degradation, or scalability barriers. Subtypes: 6.1 Dependency Violation; 6.2 Pattern Degradation; 6.3 Scalability Barrier. TD-5 is module/component scope; TD-6 is system or cross-module scope.
- TD-7 Testing Debt: insufficient or fragile quality assurance. Subtypes: 7.1 Coverage Gap; 7.2 Test Fragility; 7.3 Assertion Poverty.
- TD-8 Documentation Debt: missing or inaccurate documentation, comments, or knowledge transfer. Subtypes: 8.1 Documentation Absence; 8.2 Documentation Inaccuracy; 8.3 Naming Obscurity.
- TD-9 Infrastructure Debt: build, dependency, configuration, or deployment problems affecting development or operational reliability. Subtypes: 9.1 Dependency Problem; 9.2 Configuration Smell; 9.3 Build Fragility.
- TD-10 Security Debt: insecure coding patterns that introduce security risk. Subtypes: 10.1 Injection Surface; 10.2 Access Control Gap; 10.3 Exposure Risk.
- TD-11 Style Debt: code formatting or idiom violations of established repository conventions. Subtypes: 11.1 Formatting Inconsistency; 11.2 Idiom Violation; 11.3 Import Organization.

Change nature (fill only when is_td=true):
- DEFECT_REPAIR: the AI code was changed because it contained a bug.
- QUALITY_CLEANUP: the AI code was changed because its quality, structure, or foresight was inadequate, including refactoring, boundary handling, shared-method extraction, formatting standardization, or adaptation to a new abstraction.
- EVOLUTION: the repository gained functionality without changing the AI code (is_td=false).
- BEAUTIFICATION: formatting or lint cleanup changed AI code whose formatting was not itself problematic (is_td=false).
- UNRELATED: the fix changed an unrelated region in the same file (is_td=false).

Return only one JSON object, without Markdown fences:
{
  "touched_ai_code": true | false,
  "touched_evidence": "Explain the exact fix-diff and AI-PR patch lines that overlap, or why there is no overlap.",
  "is_td": true | false | "UNKNOWN",
  "nature": "DEFECT_REPAIR" | "QUALITY_CLEANUP" | "EVOLUTION" | "BEAUTIFICATION" | "UNRELATED" | null,
  "td_category": "TD-1" | "TD-2" | "TD-3" | "TD-4" | "TD-5" | "TD-6" | "TD-7" | "TD-8" | "TD-9" | "TD-10" | "TD-11" | "NON_TD" | null,
  "td_subtype": "e.g., 2.1 Signature Mismatch, or null",
  "confidence": 0.0,
  "reasoning": "Explain the is_td decision and the category, nature, and subtype assignment; if UNKNOWN, explain the conflicting evidence."
}

When is_td is UNKNOWN, nature and td_category must be null.
"""


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_enriched(repos=None, include_additive_near=False) -> List[Dict]:
    """Load enriched cases. If include_additive_near=True, also load PURE_ADDITIVE_NEAR from raw SZZ."""
    repos = repos or ALL_REPOS
    all_cases = []
    for repo_name in repos:
        data_file = OUTPUT_DIR / repo_name / "data" / "03_szz_enriched.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)
            all_cases.extend(cases)

        if include_additive_near:
            trace_file = OUTPUT_DIR / repo_name / "data" / "02_szz_traceability.json"
            if trace_file.exists():
                with open(trace_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                near_cases = [c for c in raw if c.get("verdict_level") == "PURE_ADDITIVE_NEAR"]
                all_cases.extend(near_cases)

    return all_cases


def group_by_pr(cases: List[Dict]) -> Dict[str, List[Dict]]:
    """Group cases by (repo, pr_number)."""
    groups = defaultdict(list)
    for c in cases:
        key = f"{c['repo_name'].split('/')[-1]}#{c['pr_number']}"
        groups[key].append(c)
    return dict(groups)


def group_by_fix_commit(pr_cases: List[Dict]) -> Dict[str, List[Dict]]:
    """Within a PR group, sub-group cases by fix commit SHA."""
    groups = defaultdict(list)
    for c in pr_cases:
        groups[c['commit_sha']].append(c)
    return dict(groups)


def extract_pr_title(case: Dict) -> str:
    """Get PR title, falling back to pr_body first line."""
    title = case.get('pr_title', '') or ''
    if not title:
        body = case.get('pr_body', '') or ''
        if body:
            first_line = body.split('\n')[0].strip()
            title = first_line[2:] if first_line.startswith('# ') else first_line[:150]
    return title or 'Unknown'


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_pr_overview(pr_cases: List[Dict], blind: bool = False) -> str:
    """Build the PR overview section.

    Classify later fixes against AI-authored pull-request code and label consequential technical debt.
    """
    c0 = pr_cases[0]
    title = extract_pr_title(c0)
    agent = c0.get('agent', '?')
    merged = c0.get('pr_merged_at', '?')
    repo = c0['repo_name'].split('/')[-1]
    pr_num = c0['pr_number']

    parts = []
    parts.append(f"## PR OVERVIEW: {repo}#{pr_num}")
    parts.append(f"Title: {title}")
    parts.append(f"Agent: {agent} | Merged: {merged}")
    parts.append(f"Files changed in PR: {len(pr_cases)}")

    body = c0.get('pr_body', '') or ''
    if body:
        if len(body) > 1500:
            body = body[:1500] + "\n... [truncated]"
        parts.append(f"\nPR Description:\n{body}")

    parts.append(f"\n### Files changed:")
    seen_files = []
    for c in pr_cases:
        fpath = c.get('filepath', '?')
        if fpath not in seen_files:
            seen_files.append(fpath)

    if blind:
        # Implementation note.
        for fpath in seen_files:
            parts.append(f"  - {fpath}")
    else:
        for fpath in seen_files:
            verdict = next((c.get('verdict_level', 'SZZ_PASSED') for c in pr_cases
                            if c.get('filepath') == fpath), 'SZZ_PASSED')
            parts.append(f"  - {fpath} [{verdict}]")

    return "\n".join(parts)


def _extract_ai_added_lines(pr_patch: str) -> set:
    """Extract the set of lines that the AI PR added (lines starting with '+' in unified diff)."""
    lines = set()
    for line in pr_patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            # Strip the '+' prefix and normalize whitespace for fuzzy matching
            stripped = line[1:].strip()
            if stripped:
                lines.add(stripped)
    return lines


def _annotate_fix_diff_with_overlap(fix_diff_text: str, ai_lines: set) -> str:
    """Annotate fix diff hunks: mark lines that overlap with AI-added code."""
    if not ai_lines:
        return fix_diff_text

    annotated = []
    for line in fix_diff_text.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            stripped = line[1:].strip()
            if stripped and stripped in ai_lines:
                annotated.append(line + '  ← [OVERLAPS WITH AI CODE]')
                continue
        annotated.append(line)
    return '\n'.join(annotated)


def _extract_overlapping_hunks(pr_patch: str, fix_diff_text: str) -> str:
    """Extract only the hunks from fix diff that overlap with AI-introduced code.

    Returns a focused view showing just the key evidence, with annotated overlap lines.
    """
    ai_lines = _extract_ai_added_lines(pr_patch)
    if not ai_lines:
        return ""

    # Parse fix diff into hunks
    hunks = []
    current_hunk = []
    in_hunk = False
    for line in fix_diff_text.split('\n'):
        if line.startswith('@@'):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]  # hunk header
            in_hunk = True
        elif in_hunk:
            current_hunk.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    # Filter: keep only hunks where at least one '-' line overlaps with AI code
    overlapping_hunks = []
    for hunk in hunks:
        has_overlap = False
        for line in hunk:
            if line.startswith('-') and not line.startswith('---'):
                stripped = line[1:].strip()
                if stripped and stripped in ai_lines:
                    has_overlap = True
                    break
        if has_overlap:
            overlapping_hunks.append(hunk)

    if not overlapping_hunks:
        return "\n[NO OVERLAP] Fix diff does not overlap AI-authored lines.\n"

    # Build output with annotations
    parts = []
    parts.append(f"\n### STEP 1 (KEY EVIDENCE): Overlapping changes — {len(overlapping_hunks)} hunk(s) where Fix touched AI code")
    parts.append("Lines marked '← [OVERLAPS WITH AI CODE]' are where AI's code was modified/deleted.\n")

    for hunk in overlapping_hunks:
        for line in hunk:
            if line.startswith('-') and not line.startswith('---'):
                stripped = line[1:].strip()
                if stripped and stripped in ai_lines:
                    parts.append(line + '  ← [OVERLAPS WITH AI CODE]')
                    continue
            parts.append(line)
        parts.append('')  # blank line between hunks

    return '\n'.join(parts)


def _format_additive_near_hunks(fix_diff_text: str) -> str:
    """Format fix diff hunks for PURE_ADDITIVE_NEAR cases — show additions near AI code."""
    if not fix_diff_text:
        return ""

    # Parse into hunks
    hunks = []
    current_hunk = []
    for line in fix_diff_text.split('\n'):
        if line.startswith('@@'):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
    if current_hunk:
        hunks.append(current_hunk)

    parts = []
    parts.append(f"\n### STEP 1 (KEY EVIDENCE): PURE_ADDITIVE_NEAR — Fix added code near AI-introduced code ({len(hunks)} hunk(s))")
    parts.append("⚠️ The fix did NOT directly delete/modify AI's code lines, but added new code nearby.")
    parts.append("This is the primary detection point for TD-3 Defensive Debt (AI's happy-path bias).\n")
    parts.append("**Check each hunk: are the additions defensive wrapping around AI's code?**")
    parts.append("  - null/undefined checks before AI code → TD-3.1 Error Handling Gap")
    parts.append("  - try/catch wrapping AI code → TD-3.1 Error Handling Gap")
    parts.append("  - edge case / boundary handling → TD-3.2 Edge Case Omission")
    parts.append("  - input validation before AI code → TD-3.1 Error Handling Gap")
    parts.append("  - concurrency/lifecycle guards → TD-3.3 State Fragility")
    parts.append("  - unrelated new functionality added nearby → NOT TD\n")

    for hunk in hunks:
        for line in hunk:
            if line.startswith('+') and not line.startswith('+++'):
                parts.append(line + '  ← [ADDED NEAR AI CODE]')
            else:
                parts.append(line)
        parts.append('')

    return '\n'.join(parts)


def build_fix_context(fix_sha: str, fix_cases: List[Dict], blind: bool = False) -> str:
    """Build context for a single fix commit affecting this PR.

    Ordering: overlapping hunks / additive-near evidence FIRST,
    then full AI PR patch and Fix diff for context.
    """
    c0 = fix_cases[0]
    msg = c0.get('commit_message', '?')
    date = c0.get('commit_date', '?')
    latency = c0.get('fix_latency_days', '?')

    parts = []
    parts.append(f"\n{'─'*50}")
    parts.append(f"## FIX COMMIT: {fix_sha[:10]}")
    parts.append(f"Message: {msg}")
    parts.append(f"Date: {date} | Latency: {latency} days after PR merge")

    latency_bucket = compute_latency_bucket(latency)
    if latency_bucket != "unknown":
        parts.append(f"Latency bucket: {latency_bucket}")

    parts.append(f"\nFiles from this PR touched by this fix ({len(fix_cases)}):")

    for c in fix_cases:
        fpath = c.get('filepath', '?')
        verdict = c.get('verdict_level', 'SZZ_PASSED')
        fix_diff = c.get('fix_diff', {})
        additions = fix_diff.get('additions', 0)
        deletions = fix_diff.get('deletions', 0)

        if blind:
            parts.append(f"\n  File: {fpath}")
        else:
            parts.append(f"\n  File: {fpath} (+{additions}/-{deletions} lines) [{verdict}]")

        pr_patch = c.get('pr_file_patch', '')
        fix_text = fix_diff.get('diff', '')

        # Compute overlap for annotation
        ai_lines = _extract_ai_added_lines(pr_patch) if pr_patch else set()
        overlap_hunks = _extract_overlapping_hunks(pr_patch, fix_text) if (pr_patch and fix_text) else ""

        # Implementation note.
        if not blind:
            has_overlap = overlap_hunks and "[NO OVERLAP]" not in overlap_hunks
            if has_overlap:
                # Normal case: fix deleted AI code lines → show overlapping hunks
                parts.append(overlap_hunks)
            elif verdict == "PURE_ADDITIVE_NEAR" and fix_text:
                # PURE_ADDITIVE_NEAR: fix added code near AI code, didn't delete AI lines
                # This is where TD-3 Defensive Debt would be detected
                additive_view = _format_additive_near_hunks(fix_text)
                if additive_view:
                    parts.append(additive_view)
            elif overlap_hunks:
                # [NO OVERLAP] but not additive near — just show the notice
                parts.append(overlap_hunks)

        # ── Step 2: Show AI PR patch (full context) ──
        if pr_patch:
            if len(pr_patch) > 6000:
                pr_patch = (pr_patch[:3500]
                            + "\n... [TRUNCATED: PR patch too long, middle omitted] ...\n"
                            + pr_patch[-2500:])
            parts.append(f"\n  ### STEP 2: Full AI PR patch (for context):")
            parts.append(f"  ```diff\n{pr_patch}\n  ```")

        # Implementation note.
        if fix_text:
            if len(fix_text) > 8000:
                fix_text = (fix_text[:4500]
                            + "\n... [TRUNCATED: fix diff too long, middle omitted] ...\n"
                            + fix_text[-3500:])
            if blind:
                parts.append(f"\n  ### STEP 3: Full Fix diff (for context):")
                parts.append(f"  ```diff\n{fix_text}\n  ```")
            else:
                annotated = _annotate_fix_diff_with_overlap(fix_text, ai_lines)
                parts.append(f"\n  ### STEP 3: Full Fix diff (for context):")
                parts.append(f"  ```diff\n{annotated}\n  ```")

    issues = c0.get('linked_issues', [])
    if issues:
        parts.append(f"\n  Linked issues:")
        for iss in issues[:3]:
            parts.append(f"    - #{iss['number']}: {iss.get('title', '')[:150]}")

    return "\n".join(parts)


def build_pr_group_prompt(pr_cases: List[Dict], blind: bool = False) -> List[Dict]:
    """Build prompts for each fix commit in a PR group.

    Classify later fixes against AI-authored pull-request code and label consequential technical debt.
    """
    fix_groups = group_by_fix_commit(pr_cases)
    sorted_fixes = sorted(fix_groups.items(), key=lambda x: x[1][0].get('commit_date', '9999'))

    prompts = []
    for fix_sha, fix_cases in sorted_fixes:
        prompt = []
        prompt.append(build_pr_overview(pr_cases, blind=blind))
        prompt.append(build_fix_context(fix_sha, fix_cases, blind=blind))

        if len(sorted_fixes) > 1:
            prompt.append(f"\n{'─'*50}")
            prompt.append("## FULL TIMELINE (all fix commits touching this PR's files):")
            for fsha, fcases in sorted_fixes:
                fmsg = fcases[0].get('commit_message', '?')[:100]
                fdate = fcases[0].get('commit_date', '?')
                n_files = len(fcases)
                is_current = "<< CURRENT" if fsha == fix_sha else ""
                prompt.append(f"  {fdate} | {fsha[:10]} | {fmsg} | {n_files} files {is_current}")

        prompts.append({
            'fix_sha': fix_sha,
            'fix_cases': fix_cases,
            'prompt': "\n".join(prompt),
            'pr_key': f"{fix_cases[0]['repo_name'].split('/')[-1]}#{fix_cases[0]['pr_number']}",
        })

    return prompts


# ═══════════════════════════════════════════════════════════════════════════
# LLM CALL
# ═══════════════════════════════════════════════════════════════════════════

def call_llm(system: str, user: str, model: str = "gpt-4o-mini-2024-07-18", max_retries: int = 3) -> Optional[Dict]:
    """Call LLM with retry logic, return parsed JSON dict or None."""
    import urllib.request
    import random

    base_url = os.environ.get("OPENAI_BASE_URL", os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))

    api_keys = []
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
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "4000")),
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    last_error = None

    for attempt in range(max_retries):
        api_key = random.choice(api_keys)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "120"))
            resp = urllib.request.urlopen(req, timeout=request_timeout)
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                return json.loads(m.group())
            return None
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(min(2.0 ** attempt, 90.0))
                continue

    print(f"  LLM Error (after {max_retries} retries): {last_error}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PRE-FILTER (Stage 0 — mechanical exclusion)
# ═══════════════════════════════════════════════════════════════════════════

def hard_prefilter(fix_cases: List[Dict]) -> Optional[Dict]:
    """Apply deterministic prefilters; return None when LLM review is needed."""
    return None


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def classify_pr_groups(pr_groups: Dict[str, List[Dict]], model: str = "gpt-4o-mini-2024-07-18",
                       workers: int = 5, checkpoint_file: Optional[str] = None,
                       blind: bool = False, retry_errors: bool = False) -> List[Dict]:
    """Classify PR groups concurrently and persist resumable checkpoints."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results = []
    results_lock = threading.Lock()

    # Implementation note.
    done_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    is_error = bool(rec.get('_llm_error'))
                    if not (retry_errors and is_error):
                        results.append(rec)
                        done_ids.add(rec.get('case_id', ''))
                except Exception:
                    continue
        if done_ids:
            print(f"[checkpoint] Resuming with {len(done_ids)} completed cases\n")

    # Implementation note.
    all_prompts = []
    for pr_key, pr_cases in sorted(pr_groups.items()):
        prompts = build_pr_group_prompt(pr_cases, blind=blind)
        for p in prompts:
            case_ids = [c.get('case_id') for c in p['fix_cases']]
            if case_ids and all(cid in done_ids for cid in case_ids):
                continue
            all_prompts.append((pr_key, p))

    total = len(all_prompts)
    print(f"Total: {len(pr_groups)} PRs, {total} fix commits queued for LLM classification\n")

    # Implementation note.
    completed = [0]
    completed_lock = threading.Lock()

    def classify_one(pr_key, p):
        fix_short = p['fix_sha'][:8]
        short_pr = pr_key[-50:] if len(pr_key) > 50 else pr_key

        try:
            max_retries = int(os.environ.get("LLM_MAX_RETRIES", "15"))
            llm_result = call_llm(SYSTEM_PROMPT, p['prompt'], model=model,
                                  max_retries=max_retries)
            if llm_result:
                for c in p['fix_cases']:
                    c['_llm_is_td'] = llm_result.get('is_td')
                    c['_llm_nature'] = llm_result.get('nature')
                    c['_llm_td_category'] = llm_result.get('td_category')
                    c['_llm_td_subtype'] = llm_result.get('td_subtype')
                    c['_llm_confidence'] = llm_result.get('confidence')
                    c['_llm_reasoning'] = llm_result.get('reasoning', '')
                td_val = llm_result.get('is_td')
                if td_val == 'UNKNOWN':
                    td_label = 'UNKNOWN'
                elif td_val:
                    td_label = 'TD'
                else:
                    td_label = 'NOT'
                conf = llm_result.get('confidence', 0)
                cat = llm_result.get('td_category') or '?'
                status = f"-> {td_label} {cat} ({conf:.0%})"
            else:
                for c in p['fix_cases']:
                    c['_llm_is_td'] = None
                    c['_llm_error'] = 'no_response'
                status = "-> ERROR"
        except Exception as e:
            for c in p['fix_cases']:
                c['_llm_is_td'] = None
                c['_llm_error'] = str(e)
            status = f"-> ERROR: {e}"

        with completed_lock:
            completed[0] += 1
            print(f"  [{completed[0]}/{len(all_prompts)}] {short_pr} @{fix_short}... {status}")

        with results_lock:
            results.extend(p['fix_cases'])
            if checkpoint_file:
                with open(checkpoint_file, 'a', encoding='utf-8') as cf:
                    for c in p['fix_cases']:
                        cf.write(json.dumps(c, ensure_ascii=False) + '\n')

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_one, pr_key, p): (pr_key, p)
                   for pr_key, p in all_prompts}
        for future in as_completed(futures):
            future.result()

    print(f"\n  [{len(all_prompts)} LLM requests, {workers} workers]")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_latency_bucket(latency_days) -> str:
    """Map a non-negative, finite latency value to its reporting bucket."""
    if isinstance(latency_days, bool) or not isinstance(latency_days, (int, float)):
        return "unknown"

    days = float(latency_days)
    if not isfinite(days) or days < 0:
        return "unknown"
    if days <= 2:
        return "immediate (0-2 days)"
    if days <= 30:
        return "short (3-30 days)"
    if days <= 180:
        return "medium (31-180 days)"
    return "long (>180 days)"


def print_stats(results: List[Dict]):
    """Print classification totals and breakdowns."""
    td = sum(1 for r in results if r.get('_llm_is_td') is True)
    not_td = sum(1 for r in results if r.get('_llm_is_td') is False)
    unknown = sum(1 for r in results if r.get('_llm_is_td') == 'UNKNOWN')
    errors = sum(1 for r in results if r.get('_llm_is_td') is None)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"RESULTS: {total} cases")
    print(f"  TD:      {td} ({td/max(1,total)*100:.1f}%)")
    print(f"  NOT TD:  {not_td} ({not_td/max(1,total)*100:.1f}%)")
    print(f"  UNKNOWN: {unknown} ({unknown/max(1,total)*100:.1f}%)")
    print(f"  Errors:  {errors}")

    # By category (TD only)
    cat_counts = Counter(r.get('_llm_td_category', '?') for r in results if r.get('_llm_is_td') is True)
    if cat_counts:
        print(f"\n  TD categories:")
        for cat, count in cat_counts.most_common():
            print(f"    {cat}: {count}")

    # By nature (TD only)
    nature_counts = Counter(r.get('_llm_nature', '?') for r in results if r.get('_llm_is_td') is True)
    if nature_counts:
        print(f"\n  TD nature:")
        for nat, count in nature_counts.most_common():
            print(f"    {nat}: {count}")

    # By repo
    by_repo = defaultdict(list)
    for r in results:
        by_repo[(r.get('repo_name') or '?').split('/')[-1]].append(r)
    print(f"\n  Per repo:")
    for repo in sorted(by_repo):
        cases = by_repo[repo]
        repo_td = sum(1 for r in cases if r.get('_llm_is_td') is True)
        repo_unk = sum(1 for r in cases if r.get('_llm_is_td') == 'UNKNOWN')
        extra = f', {repo_unk} UNKNOWN' if repo_unk else ''
        print(f"    {repo}: {len(cases)} cases, {repo_td} TD "
              f"({repo_td/max(1,len(cases))*100:.0f}%){extra}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD Classifier (Simplified) — Pure TD Judgment")
    parser.add_argument("--repo", help="Single repo name")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--limit-pr", type=int, default=0, help="Limit number of PRs (by sorted key)")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="Randomly sample N cases from all repos (includes SZZ_PASSED + PURE_ADDITIVE_NEAR)")
    parser.add_argument("--load-sample", type=str, default="", metavar="FILE",
                        help="Load pre-sampled cases from a JSON file (e.g. stratified_sample_250.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts for first 2 PRs without calling LLM")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent LLM workers (default: 5)")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Checkpoint file path (jsonl) for resume support")
    parser.add_argument("--blind", action="store_true",
                        help="Blind labeling: strip channel shape cues (verdict, +N/-M, "
                             "PURE_ADDITIVE_NEAR mapping, OVERLAPS annotation) from prompts")
    parser.add_argument("--rerun-err", type=str, default="", metavar="FILE",
                        help="Rerun cases with no LLM verdict (is_td is None) from a previous "
                             "result JSON; forces blind=True")
    args = parser.parse_args()

    if args.rerun_err:
        with open(args.rerun_err, 'r', encoding='utf-8') as f:
            prev = json.load(f)
        cases = [r for r in prev if r.get('_llm_is_td') is None]
        print(f"Rerunning {len(cases)} failed cases from {args.rerun_err}")
        if not cases:
            sys.exit(0)
    elif args.load_sample:
        sample_file = OUTPUT_DIR / args.load_sample
        print(f"Loading pre-sampled cases from {sample_file}...")
        with open(sample_file, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cases")
    else:
        repos = [args.repo] if args.repo else ALL_REPOS
        print(f"Loading cases from {len(repos)} repos...")
        # Implementation note.
        cases = load_all_enriched(repos, include_additive_near=False)
        print(f"Loaded {len(cases)} cases from enriched SZZ data")

    # ── Random sampling (case-level, not PR-level) — only when NOT using --load-sample ──
    if args.sample > 0 and not args.load_sample:
        import random
        random.seed(42)
        sample_n = min(args.sample, len(cases))
        cases = random.sample(cases, sample_n)
        print(f"Random sampled {len(cases)} cases")

    pr_groups = group_by_pr(cases)
    print(f"Grouped into {len(pr_groups)} PRs")

    sizes = Counter(len(v) for v in pr_groups.values())
    if sizes:
        print(f"Files per PR: min={min(sizes)}, max={max(sizes)}, "
              f"avg={sum(k*v for k,v in sizes.items())/len(pr_groups):.1f}")

    if args.limit_pr > 0:
        pr_keys = sorted(pr_groups.keys())[:args.limit_pr]
        pr_groups = {k: pr_groups[k] for k in pr_keys}
        print(f"Limited to {len(pr_groups)} PRs")

    model_slug = args.model.replace('/', '_')

    if args.dry_run:
        for pr_key in sorted(pr_groups.keys())[:2]:
            prompts = build_pr_group_prompt(pr_groups[pr_key], blind=args.blind)
            for p in prompts[:1]:
                print(f"\n{'='*70}")
                print(f"PR: {pr_key} | Fix: {p['fix_sha'][:10]} | blind={args.blind}")
                print(f"{'='*70}")
                print(p['prompt'][:5000])
                print(f"\n... [{len(p['prompt'])} chars total]")
    else:
        results = classify_pr_groups(pr_groups, model=args.model, workers=args.workers,
                                     checkpoint_file=args.checkpoint or None,
                                     blind=args.blind or bool(args.rerun_err))

        # Implementation note.
        if args.rerun_err:
            results_path = OUTPUT_DIR / f"_intermediate/_td_classifier_blind_rerun_{model_slug}.json"
        elif args.load_sample:
            base = args.load_sample.replace('.json', '').replace('stratified_', '')
            results_path = OUTPUT_DIR / f"_td_classifier_{base}_{model_slug}.json"
        elif args.sample > 0:
            results_path = OUTPUT_DIR / f"_td_classifier_simple_sample{args.sample}_{model_slug}.json"
        else:
            if args.blind:
                # --repo blind runs are saved per-repo so sequential runs don't overwrite each other
                repo_tag = f"_{args.repo}" if args.repo else ""
                results_path = OUTPUT_DIR / f"_td_classifier_blind{repo_tag}_{model_slug}.json"
            else:
                # --repo runs are saved per-repo so sequential runs don't overwrite each other
                repo_tag = f"_{args.repo}" if args.repo else ""
                results_path = OUTPUT_DIR / f"_td_classifier_full{repo_tag}_{model_slug}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {results_path}")

        print_stats(results)

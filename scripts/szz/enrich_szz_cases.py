"""
SZZ Case Enrichment Script
============================
For each SZZ_PASSED case, fetch full context needed for TD judgment:
  1. PR body (description + intent) from GitHub API
  2. PR file diffs (what AI wrote) from GitHub API
  3. Fix commit diff (what human changed) from local git
  4. Linked issues (referenced from PR body + commit message) from GitHub API

Output: 03_szz_enriched.json — SZZ_PASSED cases with complete context for Phase 3.

Usage:
  python src/phase2_szz/enrich_szz_cases.py --repo crewAI
  python src/phase2_szz/enrich_szz_cases.py --all
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Set, Tuple
from collections import defaultdict

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
REPO_BASE = PROJECT_ROOT / "repositories"
OUTPUT_DIR = SRC_DIR / "output"
FINAL_PRS = SRC_DIR / "output" / "repo_selection" / "final_ai_prs.json"

# ── Env ──
for p in [Path(".env"), PROJECT_ROOT / ".env"]:
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        break

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# ── GitHub API helpers ──
_api_cache: Dict[str, Any] = {}


def github_api(path: str, timeout: int = 15) -> Optional[dict]:
    """Call GitHub REST API with caching and 403/429 backoff retry."""
    url = f"https://api.github.com{path}"
    if url in _api_cache:
        return _api_cache[url]

    for attempt in range(5):
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "szz-enricher/1.0",
        })
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            _api_cache[url] = data
            return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < 4:
                wait = 20 * (2 ** attempt)
                print(f"  [RATE] HTTP {e.code} {path}: backoff {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"  [API WARN] {path}: HTTP {e.code}", flush=True)
            _api_cache[url] = None
            return None
        except Exception as e:
            print(f"  [API WARN] {path}: {e}", flush=True)
            _api_cache[url] = None
            return None
    _api_cache[url] = None
    return None


# ── Offline PR meta cache (rebuilt by prefetch_pr_meta.py; all 4,596 PRs) ──
CACHE_FILE = OUTPUT_DIR / "repo_selection" / "pr_meta_cache.json"
_pr_meta_cache: Optional[Dict[str, Any]] = None


def _load_pr_meta_cache() -> Dict[str, Any]:
    global _pr_meta_cache
    if _pr_meta_cache is None:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _pr_meta_cache = json.load(f)
        else:
            _pr_meta_cache = {}
        print(f"  [cache] loaded {len(_pr_meta_cache)} PR meta entries from {CACHE_FILE.name}")
    return _pr_meta_cache


def get_pr_details(owner: str, repo: str, pr_number: int) -> dict:
    """Get PR title/body/merge info — offline from pr_meta_cache.json."""
    entry = _load_pr_meta_cache().get(f"{owner}/{repo}#{pr_number}")
    if entry is None:
        print(f"  [cache] MISS {owner}/{repo}#{pr_number}", flush=True)
        return {}
    files = entry.get("files") or []
    additions = sum(f.get("additions", 0) for f in files)
    deletions = sum(f.get("deletions", 0) for f in files)
    return {
        "title": entry.get("title", ""),
        "body": entry.get("body", "") or "",
        "merged_at": entry.get("merged_at", ""),
        "merge_commit_sha": entry.get("merge_sha", ""),
        "html_url": f"https://github.com/{owner}/{repo}/pull/{pr_number}",
        "additions": additions,
        "deletions": deletions,
        "changed_files": len(files),
    }


def get_pr_files(owner: str, repo: str, pr_number: int) -> List[dict]:
    """Get PR file list with patches — offline from pr_meta_cache.json."""
    entry = _load_pr_meta_cache().get(f"{owner}/{repo}#{pr_number}")
    if entry is None:
        return []
    files = []
    for f in entry.get("files") or []:
        files.append({
            "filename": f.get("filename", ""),
            "status": f.get("status", ""),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
            "patch": f.get("patch", ""),
        })
    return files


def get_issue(owner: str, repo: str, issue_number: int) -> Optional[dict]:
    """Get issue details from GitHub."""
    data = github_api(f"/repos/{owner}/{repo}/issues/{issue_number}")
    if not data or "pull_request" in data:
        return None  # skip PRs (only real issues)
    return {
        "number": data.get("number"),
        "title": data.get("title", ""),
        "body": data.get("body", "") or "",
        "state": data.get("state", ""),
        "created_at": data.get("created_at", ""),
        "html_url": data.get("html_url", ""),
        "labels": [l["name"] for l in data.get("labels", [])],
    }


# ── Git helpers ──
def run_git(repo_dir: Path, args: List[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            ["git"] + args, cwd=str(repo_dir), capture_output=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def get_fix_commit_diff(repo_dir: Path, fix_sha: str, filepath: str) -> dict:
    """Get the full diff of a fix commit for a specific file."""
    diff = run_git(repo_dir, ["diff", f"{fix_sha}^..{fix_sha}", "--", filepath])
    stat = run_git(repo_dir, ["show", "--stat", "--format=", fix_sha])

    # Parse stat for +/- counts
    adds, dels = 0, 0
    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            dels += 1

    return {
        "additions": adds,
        "deletions": dels,
        "total_churn": adds + dels,
        "diff": diff[:50000],  # Truncate at 50KB for sanity
        "diff_truncated": len(diff) > 50000,
    }


def get_fix_commit_full_info(repo_dir: Path, fix_sha: str) -> dict:
    """Get full commit metadata (author, date, full message, full diff stat)."""
    format_str = "%H|%an|%ae|%ad|%s"
    log = run_git(repo_dir, ["log", "-1", f"--format={format_str}", "--date=iso", fix_sha])
    full_body = run_git(repo_dir, ["log", "-1", "--format=%B", fix_sha])
    full_diff_stat = run_git(repo_dir, ["show", "--stat", "--format=", fix_sha])

    if not log:
        return {}

    parts = log.split("|", 4)
    return {
        "sha": parts[0] if len(parts) > 0 else "",
        "author_name": parts[1] if len(parts) > 1 else "",
        "author_email": parts[2] if len(parts) > 2 else "",
        "author_date": parts[3] if len(parts) > 3 else "",
        "subject": parts[4] if len(parts) > 4 else "",
        "full_message": full_body[:2000],
        "full_diff_stat": full_diff_stat[:3000],
    }


# ── Issue reference extraction ──
def extract_issue_refs(text: str) -> List[int]:
    """Extract issue numbers from text like 'fixes #123', 'closes #456', '#789'."""
    # Pattern: keyword + #number (case insensitive)
    patterns = [
        r'(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?|ref(?:erence)?s?|addresses?|related\s+to|see|re|part\s+of)\s+#(\d+)',
        r'(?:^|\s)#(\d+)',  # bare #number references
    ]
    issues: Set[int] = set()
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            try:
                n = int(m)
                if 1 <= n <= 1000000:  # reasonable range
                    issues.add(n)
            except ValueError:
                pass
    return sorted(issues)


# ── Fix commit classification ──
def classify_fix_commit(commit_message: str) -> str:
    """Heuristic classification of fix commit intent."""
    msg_lower = commit_message.lower().strip()

    # Release / version merge
    if re.match(r'^(release|version|v\d+\.\d+|merge\s+(branch|pull|release))', msg_lower):
        return "RELEASE"

    # Bugfix patterns
    if re.match(r'^(fix|bugfix|hotfix|patch|repair|correct)', msg_lower):
        return "BUGFIX"
    if re.search(r'\b(fix|bug|issue|broken|wrong|incorrect|crash|error)\b', msg_lower):
        return "BUGFIX"

    # Feature patterns
    if re.match(r'^(feat|feature|add|implement|introduce|support)', msg_lower):
        return "FEATURE"

    # Refactor / chore patterns
    if re.match(r'^(refactor|chore|cleanup|improve|optimize|style|lint|format|type)', msg_lower):
        return "REFACTOR"

    # Revert patterns
    if re.match(r'^revert', msg_lower):
        return "REVERT"

    # Test patterns
    if re.match(r'^(test|tests|testing)', msg_lower):
        return "TEST"

    # Docs patterns
    if re.match(r'^(docs|doc|documentation|readme)', msg_lower):
        return "DOCS"

    return "OTHER"


# ── Main enrichment logic ──
def enrich_one_case(
    case: dict,
    repo_owner: str,
    repo_name: str,
    repo_dir: Path,
) -> dict:
    """Enrich a single SZZ case with PR context + fix context + issues."""
    pr_number = case["pr_number"]
    fix_sha = case["commit_sha"]
    filepath = case["filepath"]

    enriched = dict(case)  # Copy original fields

    # ── 1. PR context ──
    pr_detail = get_pr_details(repo_owner, repo_name, pr_number)
    enriched["pr_body"] = pr_detail.get("body", "")
    enriched["pr_html_url"] = pr_detail.get("html_url", "")
    enriched["pr_merged_at"] = (pr_detail.get("merged_at", "") or "")[:10]

    # Calculate fix latency (time decay)
    pr_merged_str = pr_detail.get("merged_at", "")
    fix_date_str = case.get("commit_date", "")
    if pr_merged_str and fix_date_str:
        try:
            pr_date = datetime.fromisoformat(pr_merged_str.replace("Z", "+00:00")).date()
            fix_date = datetime.strptime(fix_date_str, "%Y-%m-%d").date()
            enriched["fix_latency_days"] = (fix_date - pr_date).days
        except (ValueError, TypeError):
            enriched["fix_latency_days"] = None
    else:
        enriched["fix_latency_days"] = case.get("fix_latency_days")  # Keep if already set

    # PR files (diff)
    pr_files = get_pr_files(repo_owner, repo_name, pr_number)
    # Find the specific file's patch
    target_file_patch = ""
    all_files_summary = []
    for pf in pr_files:
        all_files_summary.append({
            "filename": pf["filename"],
            "status": pf["status"],
            "additions": pf["additions"],
            "deletions": pf["deletions"],
        })
        if pf["filename"] == filepath:
            target_file_patch = pf.get("patch", "")
    enriched["pr_files_summary"] = all_files_summary
    enriched["pr_file_patch"] = target_file_patch[:50000]  # Truncate
    enriched["pr_patch_truncated"] = len(target_file_patch) > 50000

    # ── 2. Fix commit context ──
    fix_info = get_fix_commit_full_info(repo_dir, fix_sha)
    fix_diff = get_fix_commit_diff(repo_dir, fix_sha, filepath)
    enriched["fix_commit_info"] = fix_info
    enriched["fix_diff"] = fix_diff

    # Classify fix commit
    enriched["fix_category"] = classify_fix_commit(case.get("commit_message", ""))

    # ── 3. Linked issues ──
    # Extract from PR body + commit message
    all_text = (
        (pr_detail.get("body", "") or "") + " " +
        case.get("commit_message", "") + " " +
        (fix_info.get("full_message", "") or "")
    )
    issue_refs = extract_issue_refs(all_text)

    linked_issues = []
    for issue_num in issue_refs[:5]:  # Max 5 issues per case
        issue = get_issue(repo_owner, repo_name, issue_num)
        if issue:
            linked_issues.append(issue)
        time.sleep(0.1)  # Gentle rate limit
    enriched["linked_issues"] = linked_issues
    enriched["issue_refs_found"] = issue_refs

    # ── 4. Fix is release check (quick filter) ──
    enriched["fix_is_release"] = enriched["fix_category"] == "RELEASE"

    return enriched


def load_enrichable_cases(szz_file: Path, valid_verdicts: Set[str]):
    """Load SZZ cases, keeping only enrichable verdicts.

    For large 02 files (bun/gumroad/remotion are 3.3-5.8 GB), json.load of the
    whole file OOMs (Python objects inflate ~3-5x, peak 20+ GB). Stream the
    top-level array with ijson instead and keep only the needed verdicts, so
    memory stays proportional to the enrichable subset (~100s-3000s cases).
    """
    if szz_file.stat().st_size < 500 * 1024 * 1024:
        with open(szz_file, "r", encoding="utf-8") as f:
            all_cases = json.load(f)
        passed = [c for c in all_cases if c.get("verdict_level") in valid_verdicts]
        return len(all_cases), passed

    import ijson  # required for streaming large SZZ files
    total = 0
    passed = []
    with open(szz_file, "rb") as f:
        # use_float=True: ijson parses non-int numbers as Decimal by default,
        # and json.dump cannot serialize Decimal -> TypeError on the 03 save.
        for item in ijson.items(f, "item", use_float=True):
            total += 1
            if item.get("verdict_level") in valid_verdicts:
                passed.append(item)
    return total, passed


def enrich_repo(repo_full: str) -> dict:
    """Enrich all SZZ_PASSED cases for one repo."""
    repo_name = repo_full.split("/")[1]
    repo_owner = repo_full.split("/")[0]
    repo_dir = REPO_BASE / repo_name

    if not repo_dir.exists() or not (repo_dir / ".git").exists():
        print(f"  [SKIP] Repo dir not found: {repo_dir}")
        return {"repo": repo_full, "cases_enriched": 0, "error": "repo_not_found"}

    szz_file = OUTPUT_DIR / repo_name / "data" / "02_szz_traceability.json"
    if not szz_file.exists():
        print(f"  [SKIP] No SZZ output found: {szz_file}")
        return {"repo": repo_full, "cases_enriched": 0, "error": "no_szz_output"}

    # Include PURE_ADDITIVE_NEAR: fix only adds code (no deletions) but within 50 lines of AI's code
    # These could be guard conditions, error handling, edge case checks added near AI code
    valid_verdicts = {"SZZ_PASSED", "PURE_ADDITIVE_NEAR"}
    total_cases, passed_cases = load_enrichable_cases(szz_file, valid_verdicts)
    near_count = sum(1 for c in passed_cases if c.get("verdict_level") == "PURE_ADDITIVE_NEAR")
    print(f"  {total_cases} total → {len(passed_cases)} to enrich (SZZ_PASSED={len(passed_cases)-near_count} + PURE_ADDITIVE_NEAR={near_count})")

    if not passed_cases:
        return {"repo": repo_full, "total_cases": total_cases, "cases_enriched": 0}

    enriched = []
    out_file = OUTPUT_DIR / repo_name / "data" / "03_szz_enriched.json"
    if out_file.exists():
        # Resume: skip case_ids already enriched in a previous partial run.
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                enriched = json.load(f)
            done_ids = {c.get("case_id") for c in enriched}
            todo = [c for c in passed_cases if c.get("case_id") not in done_ids]
            print(f"  resume: {len(enriched)} already enriched, {len(todo)} remaining")
            passed_cases = todo
        except Exception:
            enriched = []

    for i, case in enumerate(passed_cases):
        print(f"  [{i+1}/{len(passed_cases)}] {case['case_id']}...", end=" ", flush=True)
        try:
            rich = enrich_one_case(case, repo_owner, repo_name, repo_dir)
            enriched.append(rich)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            # Save case with error marker
            case["_enrich_error"] = str(e)
            enriched.append(case)

        # Incremental save every 50 cases (resume-safe: a kill loses <=49 cases,
        # ~40s of work; rewriting a huge 03 every 5 cases is pure serialization cost)
        if (i + 1) % 50 == 0:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)

        # PR detail + files are offline now; only issue lookups hit the API.
        time.sleep(0.1)

    # Final save
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # Summary stats
    fix_cats = defaultdict(int)
    for c in enriched:
        fix_cats[c.get("fix_category", "?")] += 1
    has_issues = sum(1 for c in enriched if c.get("linked_issues"))
    releases = sum(1 for c in enriched if c.get("fix_is_release"))

    print(f"  Saved: {out_file}")
    print(f"  Fix categories: {dict(fix_cats)}")
    print(f"  Cases with linked issues: {has_issues}/{len(enriched)}")
    print(f"  Release commits (likely false positives): {releases}/{len(enriched)}")

    return {
        "repo": repo_full,
        "total_cases": total_cases,
        "cases_enriched": len(enriched),
        "fix_categories": dict(fix_cats),
        "with_issues": has_issues,
        "release_commits": releases,
    }


# ── CLI ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enrich SZZ_PASSED cases with full context")
    parser.add_argument("--repo", default="", help="Repo slug, e.g. crewAIInc/crewAI")
    parser.add_argument("--all", action="store_true", help="Enrich all repos with SZZ output")
    parser.add_argument("--dry-run", action="store_true", help="Count SZZ_PASSED cases without enriching")
    args = parser.parse_args()

    t0 = time.time()

    if args.dry_run:
        # Just count
        print("=== Dry Run: Counting SZZ_PASSED cases ===")
        total_passed = 0
        for repo_dir in sorted(OUTPUT_DIR.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith("_"):
                continue
            szz_file = repo_dir / "data" / "02_szz_traceability.json"
            if szz_file.exists():
                with open(szz_file, "r", encoding="utf-8") as f:
                    cases = json.load(f)
                passed = sum(1 for c in cases if c.get("verdict_level") == "SZZ_PASSED")
                if passed > 0:
                    print(f"  {repo_dir.name}: {passed} SZZ_PASSED")
                    total_passed += passed
        print(f"\nTotal SZZ_PASSED across all repos: {total_passed}")
        print(f"Estimated API calls: ~{total_passed * 2} (PR detail + PR files)")
    elif args.all:
        print("=== Batch Enrichment: All Repos ===")
        all_summaries = []
        for repo_dir in sorted(OUTPUT_DIR.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith("_"):
                continue
            szz_file = repo_dir / "data" / "02_szz_traceability.json"
            if not szz_file.exists():
                continue
            with open(szz_file, "r", encoding="utf-8") as f:
                cases = json.load(f)
            passed = sum(1 for c in cases if c.get("verdict_level") == "SZZ_PASSED")
            if passed == 0:
                continue

            # Find repo_full from cases
            repo_full = cases[0].get("repo_name", "")
            if not repo_full:
                continue

            print(f"\n{'='*60}")
            print(f"Repo: {repo_full} ({passed} SZZ_PASSED)")
            print(f"{'='*60}")
            summary = enrich_repo(repo_full)
            all_summaries.append(summary)

        print(f"\n{'='*60}")
        print("Batch Enrichment Complete")
        print(f"{'='*60}")
        total = sum(s["cases_enriched"] for s in all_summaries)
        print(f"Total cases enriched: {total}")
        print(f"Total time: {time.time() - t0:.0f}s")
    elif args.repo:
        print(f"=== Enrichment: {args.repo} ===")
        summary = enrich_repo(args.repo)
        print(f"\nDone: {summary}")
    else:
        parser.print_help()

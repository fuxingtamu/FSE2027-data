"""Collect and match non-AI-attributed PR controls for the AI-PR study.

The control group is deliberately named *non-AI-attributed, non-bot* rather
than ``human-authored``.  GitHub metadata can exclude known AI-attributed PRs
and bot accounts, but cannot rule out unlabelled AI assistance by a human user.

The canonical AI analysis cohort is reconstructed from the full SZZ/classifier
input, rather than from the broader 4,469-PR descriptive roster.  This makes
the comparison eligible for the same downstream tracing pipeline: every AI PR
in the matching population has at least one traceable code member.

Matching is 1:1 without replacement, within repository, using a deterministic
greedy assignment.  A candidate must be within declared calipers for merge
date, log code churn, and log changed-file count.  The match score is the sum
of the three normalized distances.  The script writes all inputs, candidates,
matches, and diagnostics so that later SZZ and semantic classification can be
run with the identical protocol for both cohorts.

Examples
--------
Collect and match one repository as a smoke test::

    python anonymous_release/scripts/collect_and_match_controls.py --repos microsoft/vscode

Run all source repositories, reusing the persisted candidate cache::

    python anonymous_release/scripts/collect_and_match_controls.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    import ijson
except ImportError:  # pragma: no cover - fallback for a minimal environment
    ijson = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "td_classifier_cases.json"
AI_ROSTER = ROOT / "data" / "ai_prs_2717.csv"
OUT = ROOT / "results" / "matching"
COLLECTION_STRATEGY_VERSION = 6

# The historic classifier source uses some pre-transfer repository names.  The
# left-hand name remains the analysis stratum; the right-hand name is queried
# from GitHub and recorded in every output row.
REPO_ALIASES = {
    "microsoft/aspire": "microsoft/aspire",
    # GitHub currently resolves the historical cal.com name to cal.diy.  Keep
    # the canonical analysis stratum and API query on the same repository.
    "calcom/cal.diy": "calcom/cal.diy",
    "firecrawl/firecrawl": "firecrawl/firecrawl",
}

QUERY = """
query($query: String!, $cursor: String) {
  search(query: $query, type: ISSUE, first: 5, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        mergedAt
        createdAt
        additions
        deletions
        changedFiles
        isDraft
        title
        bodyText
        url
        author { __typename login }
        files(first: 30) { nodes { path additions deletions } }
      }
    }
  }
  rateLimit { remaining resetAt cost }
}
"""


def load_env() -> None:
    """Load local credentials without printing values."""
    for path in (ROOT / ".env", Path.cwd() / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip().removeprefix("export ")
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))
        return


def parse_day(value: str | None) -> date | None:
    if not value:
        return None
    value = str(value).strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def iso_day(d: date) -> str:
    return d.isoformat()


def month_windows(start: date, end: date, days: int) -> Iterable[tuple[date, date]]:
    cursor = start
    delta = timedelta(days=days - 1)
    while cursor <= end:
        upper = min(end, cursor + delta)
        yield cursor, upper
        cursor = upper + timedelta(days=1)


LANGUAGE_EXTENSIONS = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".cs": "C#", ".c": "C", ".h": "C/C++", ".cpp": "C++",
    ".php": "PHP", ".rb": "Ruby", ".swift": "Swift", ".scala": "Scala",
}
NON_CODE_SUFFIXES = {".md", ".rst", ".txt", ".json", ".yml", ".yaml", ".xml", ".toml", ".lock"}


def language_from_paths(paths: Iterable[str]) -> str:
    counts: Counter[str] = Counter()
    for raw in paths:
        path = Path(raw)
        suffix = path.suffix.lower()
        language = LANGUAGE_EXTENSIONS.get(suffix)
        if language and suffix not in NON_CODE_SUFFIXES:
            counts[language] += 1
    return counts.most_common(1)[0][0] if counts else "Unknown"


def pr_type(title: str, body: str = "") -> str:
    text = f"{title} {body}".lower()
    if any(x in text for x in ("fix", "bug", "hotfix", "regression", "error")):
        return "bug-fix"
    if any(x in text for x in ("refactor", "restructure", "cleanup", "clean-up", "reorganize", "migrate")):
        return "refactor"
    if any(x in text for x in ("feat", "feature", "implement", "support", "add ", "introduce")):
        return "feature"
    return "other"


@dataclass(frozen=True)
class PRFeatures:
    cohort: str
    repo: str
    api_repo: str
    number: int
    merged_at: str
    additions: int
    deletions: int
    changed_files: int
    code_churn: int
    primary_language: str = "Unknown"
    pr_type: str = "other"
    title: str = ""
    url: str = ""
    author_login: str = ""
    author_type: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}#{self.number}"


def iter_source() -> Iterable[dict[str, Any]]:
    if ijson is not None:
        with SOURCE.open("rb") as fh:
            yield from ijson.items(fh, "item")
    else:
        yield from json.loads(SOURCE.read_text(encoding="utf-8"))


def summarize_files(files: Any) -> tuple[int, int, int]:
    additions = deletions = changed = 0
    for item in files or []:
        additions += int(item.get("additions") or 0)
        deletions += int(item.get("deletions") or 0)
        changed += 1
    return additions, deletions, changed


def build_ai_cohort(cutoff: date) -> list[PRFeatures]:
    """Deduplicate the full analysis source to one PR-level matching row."""
    rows: dict[tuple[str, int], PRFeatures] = {}
    for item in iter_source():
        repo = str(item.get("repo_name") or "")
        number = item.get("pr_number")
        merged = parse_day(item.get("pr_merged_at"))
        if not repo or number is None or merged is None or merged > cutoff:
            continue
        key = (repo, int(number))
        if key in rows:
            continue
        adds, dels, files = summarize_files(item.get("pr_files_summary"))
        if files <= 0 or adds + dels <= 0:
            continue
        rows[key] = PRFeatures(
            cohort="AI-attributed",
            repo=repo,
            api_repo=REPO_ALIASES.get(repo, repo),
            number=int(number),
            merged_at=iso_day(merged),
            additions=adds,
            deletions=dels,
            changed_files=files,
            code_churn=adds + dels,
            primary_language=language_from_paths(
                f.get("filename", "") for f in (item.get("pr_files_summary") or [])
            ),
            pr_type=pr_type(str(item.get("pr_title") or ""), str(item.get("pr_body") or "")),
            title=str(item.get("pr_title") or ""),
            url=str(item.get("pr_html_url") or ""),
        )
    return sorted(rows.values(), key=lambda x: (x.repo, x.merged_at, x.number))


def canonical_roster_keys() -> set[tuple[str, int]]:
    """All known AI-attributed PRs, not only SZZ-eligible ones."""
    keys: set[tuple[str, int]] = set()
    with AI_ROSTER.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            repo = row.get("repo") or row.get("\ufeffrepo") or ""
            number = row.get("number")
            if repo and number and str(number).isdigit():
                keys.add((REPO_ALIASES.get(repo, repo), int(number)))
    return keys


def graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not configured")
    last_error: Exception | None = None
    response = None
    for attempt in range(4):
        try:
            response = requests.post(
                "https://api.github.com/graphql",
                headers={"Authorization": f"bearer {token}", "User-Agent": "ai-pr-human-control-study"},
                json={"query": query, "variables": variables},
                timeout=45,
            )
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError(f"GitHub GraphQL request failed after retries: {last_error}")
    if response.status_code != 200:
        raise RuntimeError(f"GitHub GraphQL HTTP {response.status_code}")
    data = response.json()
    if data.get("errors"):
        message = "; ".join(str(e.get("message", "GraphQL error")) for e in data["errors"])
        raise RuntimeError(message)
    return data["data"]


def collect_repo_candidates(
    repo: str,
    ai_rows: list[PRFeatures],
    known_ai: set[tuple[str, int]],
    cache_path: Path,
    date_caliper: int,
    query_window_days: int,
    refresh: bool,
    candidate_factor: float,
) -> list[PRFeatures]:
    """Collect candidate controls in a bounded merged-date range from GitHub."""
    if cache_path.exists() and not refresh:
        cached_rows = json.loads(cache_path.read_text(encoding="utf-8"))
        # Empty caches can be artifacts of a stale repository alias or a
        # previous failed search.  Re-query them after an alias/strategy fix.
        if cached_rows:
            return [PRFeatures(**row) for row in cached_rows]

    progress_path = cache_path.with_suffix(".progress.json")
    if refresh and progress_path.exists():
        progress_path.unlink()

    api_repo = REPO_ALIASES.get(repo, repo)
    merged_dates = [parse_day(r.merged_at) for r in ai_rows]
    start = min(d for d in merged_dates if d is not None) - timedelta(days=date_caliper)
    end = max(d for d in merged_dates if d is not None) + timedelta(days=date_caliper)
    merged: dict[int, PRFeatures] = {}
    completed_windows: set[str] = set()
    if progress_path.exists() and not refresh:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if (saved.get("strategy_version") == COLLECTION_STRATEGY_VERSION
                and saved.get("repo") == repo and saved.get("api_repo") == api_repo
                and saved.get("query_window_days") == query_window_days
                and saved.get("candidate_factor") == candidate_factor):
            merged = {int(row["number"]): PRFeatures(**row) for row in saved.get("rows", [])}
            completed_windows = set(saved.get("completed_windows", []))
            print(f"[resume] {repo}: {len(completed_windows)} windows, {len(merged)} candidates", flush=True)
    request_count = 0

    def collect_interval(lower: date, upper: date, target: int) -> int:
        """Collect up to target eligible controls in one date interval.

        GitHub Search only exposes a reliable result set when issueCount <=
        1,000.  High-volume repositories are therefore split recursively;
        the split is internal and does not alter the matching caliper.
        """
        nonlocal request_count
        if target <= 0 or lower > upper:
            return 0
        before = len(merged)
        search_query = (
            f"repo:{api_repo} is:pr is:merged "
            f"merged:{iso_day(lower)}..{iso_day(upper)}"
        )
        cursor: str | None = None
        while True:
            data = graphql(QUERY, {"query": search_query, "cursor": cursor})
            block = data["search"]
            request_count += 1
            if block["issueCount"] > 1000:
                if lower >= upper:
                    raise RuntimeError(
                        f"{api_repo} {lower}..{upper} still exceeds GitHub's 1,000-result limit"
                    )
                midpoint = lower + (upper - lower) // 2
                first = collect_interval(lower, midpoint, target)
                return first + collect_interval(midpoint + timedelta(days=1), upper, target - first)
            for node in block["nodes"]:
                if len(merged) - before >= target:
                    break
                if not node or not node.get("mergedAt"):
                    continue
                number = int(node["number"])
                author = node.get("author") or {}
                if (api_repo, number) in known_ai or author.get("__typename") == "Bot":
                    continue
                additions = int(node.get("additions") or 0)
                deletions = int(node.get("deletions") or 0)
                changed = int(node.get("changedFiles") or 0)
                if additions + deletions <= 0 or changed <= 0:
                    continue
                merged[number] = PRFeatures(
                    cohort="non-AI-attributed-non-bot",
                    repo=repo,
                    api_repo=api_repo,
                    number=number,
                    merged_at=str(node["mergedAt"])[:10],
                    additions=additions,
                    deletions=deletions,
                    changed_files=changed,
                    code_churn=additions + deletions,
                    primary_language=language_from_paths(
                        f.get("path", "") for f in (node.get("files") or {}).get("nodes", [])
                    ),
                    pr_type=pr_type(str(node.get("title") or ""), str(node.get("bodyText") or "")),
                    title=str(node.get("title") or ""),
                    url=str(node.get("url") or ""),
                    author_login=str(author.get("login") or ""),
                    author_type=str(author.get("__typename") or ""),
                )
            if len(merged) - before >= target:
                break
            page = block["pageInfo"]
            if not page["hasNextPage"]:
                break
            cursor = page["endCursor"]
            time.sleep(0.15)
        return len(merged) - before

    for lower, upper in month_windows(start, end, query_window_days):
        window_key = f"{iso_day(lower)}..{iso_day(upper)}"
        if window_key in completed_windows:
            continue
        relevant_ai = sum(
            1 for row in ai_rows
            if (parse_day(row.merged_at) is not None
                and lower <= parse_day(row.merged_at) <= upper)
        )
        # Empty windows cannot contribute a match.  This is the main reduction
        # from the previous all-PR collection strategy.
        if relevant_ai == 0:
            completed_windows.add(window_key)
            continue
        target_candidates = max(5, math.ceil(relevant_ai * candidate_factor))
        print(f"[collect] {repo}: window {window_key}", flush=True)
        collect_interval(lower, upper, target_candidates)

        completed_windows.add(window_key)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({
            "repo": repo,
            "api_repo": api_repo,
            "strategy_version": COLLECTION_STRATEGY_VERSION,
            "date_range": [iso_day(start), iso_day(end)],
            "query_window_days": query_window_days,
            "candidate_factor": candidate_factor,
            "completed_windows": sorted(completed_windows),
            "rows": [asdict(row) for row in sorted(merged.values(), key=lambda x: (x.merged_at, x.number))],
        }, ensure_ascii=False), encoding="utf-8")
        print(f"[checkpoint] {repo}: {len(completed_windows)} windows, {len(merged)} candidates", flush=True)

    rows = sorted(merged.values(), key=lambda x: (x.merged_at, x.number))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_path.exists():
        progress_path.unlink()
    print(f"[collect] {repo}: {len(rows)} eligible candidates ({request_count} GraphQL requests)")
    return rows


def log_distance(a: int, b: int) -> float:
    return abs(math.log1p(a) - math.log1p(b))


def candidate_score(
    ai: PRFeatures,
    control: PRFeatures,
    date_caliper: int,
    churn_caliper: float,
    file_caliper: float,
) -> tuple[float, dict[str, float]] | None:
    if (ai.primary_language != "Unknown" and control.primary_language != "Unknown"
            and ai.primary_language != control.primary_language):
        return None
    # PR type is inferred from public title/body text.  Unknown/other types are
    # retained but never treated as an exact match.
    if ai.pr_type != "other" and control.pr_type != "other" and ai.pr_type != control.pr_type:
        return None
    ai_date, control_date = parse_day(ai.merged_at), parse_day(control.merged_at)
    assert ai_date is not None and control_date is not None
    d_date = abs((control_date - ai_date).days)
    d_churn = log_distance(ai.code_churn, control.code_churn)
    d_files = log_distance(ai.changed_files, control.changed_files)
    if d_date > date_caliper or d_churn > churn_caliper or d_files > file_caliper:
        return None
    components = {
        "merge_day_distance": float(d_date),
        "log_churn_distance": d_churn,
        "log_files_distance": d_files,
    }
    score = d_date / date_caliper + d_churn / churn_caliper + d_files / file_caliper
    return score, components


def greedy_match(
    ai_rows: list[PRFeatures],
    candidates_by_repo: dict[str, list[PRFeatures]],
    date_caliper: int,
    churn_caliper: float,
    file_caliper: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match rare cases first, then choose the nearest unused control."""
    options: dict[str, list[tuple[float, PRFeatures, dict[str, float]]]] = {}
    for ai in ai_rows:
        opts = []
        for control in candidates_by_repo.get(ai.repo, []):
            scored = candidate_score(ai, control, date_caliper, churn_caliper, file_caliper)
            if scored is not None:
                score, components = scored
                opts.append((score, control, components))
        options[ai.key] = sorted(opts, key=lambda x: (x[0], x[1].merged_at, x[1].number))

    used: set[tuple[str, int]] = set()
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    # Cases with fewer options are assigned first; ties are deterministic.
    for ai in sorted(ai_rows, key=lambda x: (len(options[x.key]), x.repo, x.merged_at, x.number)):
        choice = next((item for item in options[ai.key] if (item[1].api_repo, item[1].number) not in used), None)
        if choice is None:
            unmatched.append({
                "ai_key": ai.key,
                "repo": ai.repo,
                "number": ai.number,
                "merged_at": ai.merged_at,
                "candidate_count_before_assignment": len(options[ai.key]),
            })
            continue
        score, control, components = choice
        used.add((control.api_repo, control.number))
        matched.append({
            "ai": asdict(ai),
            "control": asdict(control),
            "score": score,
            **components,
        })
    return matched, unmatched


def summarize_matches(matches: list[dict[str, Any]], unmatched: list[dict[str, Any]], ai_rows: list[PRFeatures], args: argparse.Namespace) -> dict[str, Any]:
    by_repo = Counter(m["ai"]["repo"] for m in matches)
    ai_churn = [m["ai"]["code_churn"] for m in matches]
    control_churn = [m["control"]["code_churn"] for m in matches]
    ai_files = [m["ai"]["changed_files"] for m in matches]
    control_files = [m["control"]["changed_files"] for m in matches]
    return {
        "unit": "matched PR pair",
        "control_label": "non-AI-attributed, non-bot PR",
        # Store only the filename so generated summaries do not expose local paths.
        "ai_source": SOURCE.name,
        "n_ai_eligible": len(ai_rows),
        "n_matched_pairs": len(matches),
        "n_unmatched_ai": len(unmatched),
        "match_rate": len(matches) / len(ai_rows) if ai_rows else 0.0,
        "calipers": {
            "same_repository": True,
            "merge_date_days": args.date_caliper_days,
            "log1p_code_churn": args.churn_caliper,
            "log1p_changed_files": args.file_caliper,
            "without_replacement": True,
        },
        "mean_ai_churn": sum(ai_churn) / len(ai_churn) if ai_churn else None,
        "mean_control_churn": sum(control_churn) / len(control_churn) if control_churn else None,
        "mean_ai_changed_files": sum(ai_files) / len(ai_files) if ai_files else None,
        "mean_control_changed_files": sum(control_files) / len(control_files) if control_files else None,
        "matches_by_source_repo": dict(sorted(by_repo.items())),
        "interpretation": (
            "Controls are non-AI-attributed and non-bot according to observable GitHub metadata. "
            "They are not verified to be free of unlabelled AI assistance."
        ),
    }


def write_csv(path: Path, matches: list[dict[str, Any]]) -> None:
    fields = [
        "ai_repo", "ai_number", "ai_merged_at", "ai_churn", "ai_files",
        "control_repo", "control_api_repo", "control_number", "control_merged_at",
        "control_churn", "control_files", "control_author_type", "score",
        "merge_day_distance", "log_churn_distance", "log_files_distance",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in matches:
            ai, control = item["ai"], item["control"]
            writer.writerow({
                "ai_repo": ai["repo"], "ai_number": ai["number"], "ai_merged_at": ai["merged_at"],
                "ai_churn": ai["code_churn"], "ai_files": ai["changed_files"],
                "control_repo": control["repo"], "control_api_repo": control["api_repo"],
                "control_number": control["number"], "control_merged_at": control["merged_at"],
                "control_churn": control["code_churn"], "control_files": control["changed_files"],
                "control_author_type": control["author_type"], "score": item["score"],
                "merge_day_distance": item["merge_day_distance"],
                "log_churn_distance": item["log_churn_distance"], "log_files_distance": item["log_files_distance"],
            })


def main() -> None:
    global SOURCE, AI_ROSTER, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="*", help="source-repo strata to collect; default: all")
    parser.add_argument("--source", type=Path, default=SOURCE, help="canonical AI classifier JSON input")
    parser.add_argument("--ai-roster", type=Path, default=AI_ROSTER, help="AI PR roster CSV used to exclude AI controls")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="directory for candidate caches and match outputs")
    parser.add_argument("--cutoff", default="2026-07-31", help="inclusive corpus cutoff (YYYY-MM-DD)")
    parser.add_argument("--date-caliper-days", type=int, default=45)
    parser.add_argument("--churn-caliper", type=float, default=0.70, help="max |log1p(churn) difference|")
    parser.add_argument("--file-caliper", type=float, default=0.70, help="max |log1p(changed-file count) difference|")
    parser.add_argument("--query-window-days", type=int, default=7, help="merged-date query window; lower it if a GitHub search exceeds 1,000 results")
    parser.add_argument("--candidate-factor", type=float, default=1.5, help="candidate-pool multiplier per AI PR in each time window")
    parser.add_argument("--refresh", action="store_true", help="ignore persisted GitHub candidate caches")
    parser.add_argument("--dry-run", action="store_true", help="rebuild matching only from existing candidate caches")
    args = parser.parse_args()

    SOURCE = args.source.resolve()
    AI_ROSTER = args.ai_roster.resolve()
    OUT = args.output_dir.resolve()

    cutoff = parse_day(args.cutoff)
    if cutoff is None:
        raise SystemExit("--cutoff must be YYYY-MM-DD")
    if not SOURCE.exists() or not AI_ROSTER.exists():
        raise SystemExit("canonical AI source or AI roster is missing")
    if (args.date_caliper_days <= 0 or args.churn_caliper <= 0 or args.file_caliper <= 0
            or args.candidate_factor < 1.0):
        raise SystemExit("all calipers must be positive")

    load_env()
    ai_rows = build_ai_cohort(cutoff)
    selected = set(args.repos or [r.repo for r in ai_rows])
    unknown = selected - {r.repo for r in ai_rows}
    if unknown:
        raise SystemExit(f"unknown source repos: {sorted(unknown)}")
    ai_rows = [r for r in ai_rows if r.repo in selected]
    known_ai = canonical_roster_keys()
    candidates_by_repo: dict[str, list[PRFeatures]] = {}
    for repo in sorted(selected):
        cache = OUT / "github_candidates" / f"{repo.replace('/', '__')}.json"
        repo_ai = [r for r in ai_rows if r.repo == repo]
        if args.dry_run and not cache.exists():
            raise SystemExit(f"--dry-run requested but cache is missing: {cache}")
        candidates_by_repo[repo] = collect_repo_candidates(
            repo, repo_ai, known_ai, cache, args.date_caliper_days,
            args.query_window_days, args.refresh,
            args.candidate_factor,
        )

    matched, unmatched = greedy_match(
        ai_rows, candidates_by_repo, args.date_caliper_days, args.churn_caliper, args.file_caliper
    )
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "all" if not args.repos else "_".join(sorted(selected)).replace("/", "__")
    (OUT / f"ai_analysis_cohort_{suffix}.json").write_text(
        json.dumps([asdict(r) for r in ai_rows], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / f"matched_pairs_{suffix}.json").write_text(
        json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / f"unmatched_ai_{suffix}.json").write_text(
        json.dumps(unmatched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(OUT / f"matched_pairs_{suffix}.csv", matched)
    summary = summarize_matches(matched, unmatched, ai_rows, args)
    (OUT / f"matching_summary_{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

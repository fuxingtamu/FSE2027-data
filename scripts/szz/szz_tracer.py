"""
SZZ Tracer — Programmatic TD lineage tracing (inspired by AgentSZZ paper).

Given an AI PR and a repo, this module:
  1. Reads AI PR metadata from final_ai_prs.json
  2. Fetches merge SHA + PR file diffs via GitHub API
  3. git log tracks all downstream commits touching the same files
  4. git blame -w -C traces deleted lines back to their introducers
  5. Applies 5-layer programmatic gating (Rules 1a-1e)
  6. Generates szz_traceability blocks consumable by TechDebtAnalyzer

Design decisions from AgentSZZ (Lyu et al., 2026):
  - Non-code file filtering (cosmetic/meta/config patterns)
  - Refactor penetration via iterative re-blame
  - Multi-source detection (deleted code comes from multiple PRs)
  - Temporal causality check (fix must post-date PR merge)
  - Weak-signal auto-intercept (base_confidence < 0.2 with few related lines)

Usage:
  python src/phase2_szz/szz_tracer.py --repo microsoft/aspire --limit 10
  python src/phase2_szz/szz_tracer.py --repo getsentry/sentry --workers 15
"""

import subprocess, json, csv, os, re, sys, time, hashlib, ctypes
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Optional, Dict, Any, List, Set, Tuple

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
REPO_BASE = PROJECT_ROOT / "repositories"
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

# ── Non-code file patterns (across languages) ──
NON_CODE_PATTERNS = [
    # Config / build
    '.json', '.yml', '.yaml', '.xml', '.toml', '.ini', '.cfg',
    '.csproj', '.props', '.targets', '.sln', '.slnx',
    'Directory.Packages.', 'Directory.Build.',
    'packages.config', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
    'Cargo.lock', 'go.sum', 'Gemfile.lock', 'Pipfile.lock',
    'CMakeLists.txt', 'Makefile', 'Dockerfile', '.dockerignore',
    '.gitignore', '.gitattributes', '.editorconfig',
    # Docs / comments
    '.md', '.rst', '.txt', 'README', 'CHANGELOG', 'CONTRIBUTING',
    'LICENSE', 'NOTICE', 'AUTHORS',
    # Generated / resource
    '.verified.', 'CompatibilitySuppressions.xml',
    '.xlf', '.resx', '.Designer.cs', '.g.cs', '.g.i.cs',
    '.razor', '.css', '.scss', '.svg', '.png', '.ico', '.woff',
    '.module.bicep', 'launchSettings.json', 'appsettings.',
    'devcontainer.json', 'tsconfig.json',
    # Lock / auto-gen
    '.snap', '.lock', 'generated', '__pycache__',
]


def is_non_code(filepath: str) -> bool:
    """Filter non-code files that shouldn't be SZZ candidates."""
    return any(p in filepath for p in NON_CODE_PATTERNS)


# ── Git helpers ──
def run_git(repo_dir: Path, args: List[str], timeout: int = 60) -> str:
    """Run git command, return stdout or ''."""
    try:
        r = subprocess.run(
            ["git"] + args, cwd=str(repo_dir), capture_output=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def run_git_lines(repo_dir: Path, args: List[str], timeout: int = 60) -> List[str]:
    """Run git command, return stdout as list of lines."""
    out = run_git(repo_dir, args, timeout)
    return out.split("\n") if out else []


# ── Ancestry memoization ──
# git merge-base --is-ancestor returns its answer as the exit code (0/1/128),
# not stdout, so it cannot go through git_memo. Memoize the boolean per pair.
_ancestor_memo: Dict[Tuple[str, str], Optional[bool]] = {}


def is_ancestor(repo_dir: Path, a: str, b: str) -> Optional[bool]:
    """True if commit `a` is an ancestor of `b` (or equal). None on git error."""
    key = (a, b)
    if key in _ancestor_memo:
        return _ancestor_memo[key]
    if not a or not b:
        _ancestor_memo[key] = None
        return None
    try:
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", a, b],
            cwd=str(repo_dir), capture_output=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
        if r.returncode == 0:
            _ancestor_memo[key] = True
        elif r.returncode == 1:
            _ancestor_memo[key] = False
        else:
            _ancestor_memo[key] = None  # 128: unknown revision / bad args
    except Exception:
        _ancestor_memo[key] = None
    return _ancestor_memo[key]


# ── GitHub API ──
_pr_meta_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()
_offline = False  # --offline: never call GitHub API, read pr_meta_cache.json only

# Pre-load persisted PR metadata (filled by prefetch_pr_meta.py) so SZZ can run offline.
_PR_META_CACHE_FILE = SRC_DIR / "output" / "repo_selection" / "pr_meta_cache.json"
if _PR_META_CACHE_FILE.exists():
    try:
        with open(_PR_META_CACHE_FILE, encoding="utf-8") as _f:
            _pr_meta_cache.update(json.load(_f))
        print(f"  [meta] loaded {len(_pr_meta_cache)} PRs from {_PR_META_CACHE_FILE.name}")
    except Exception:
        pass

# ── git-output memoization ──
# Safe: the repo is read-only during a run, so a given (rev, path) always yields
# identical git output. Memoizing skips repeated subprocess spawns (the dominant
# cost: heavy PRs run 40-100 `git blame -C` each) without changing any verdict.
_git_small: Dict[Tuple, str] = {}                        # diff / commit-subject outputs
_blame_cache: "OrderedDict[Tuple, str]" = OrderedDict()  # porcelain blame, LRU-bounded
_BLAME_LRU_MAX, _BLAME_MAX_BYTES = 1024, 256 * 1024
# A single fix commit can rewrite a whole generated/bundled file (webpack
# dist/index.js etc.) into a multi-MB / tens-of-thousands-of-line diff.
# _reblame_prior_writer re-parses that diff once per deleted line, so the cost
# is O(#deleted_lines x #diff_lines) — a 15k-line rewrite of a bundle has ~15k
# deleted lines and a 17k-line diff: ~260M pure-Python ops per fix commit, an
# hour-plus spin with zero git work (observed hang on onlook PR #1892). Above
# these bounds the re-blame is skipped and the line is conservatively refuted.
# (line-count guard catches bundle rewrites; byte guard catches single-line
# minified giants that have few newlines but huge size)
_REBLAME_MAX_DIFF = 5_000_000
_REBLAME_MAX_DIFF_LINES = 10_000


def git_memo(kind: str, args: Tuple, run) -> str:
    """Memoized read-only git call. `kind` in {"blame", "diff", "msg", "log"}."""
    key = (kind,) + args
    with _cache_lock:
        if kind == "blame":
            v = _blame_cache.get(key)
            if v is not None:
                _blame_cache.move_to_end(key)
                return v
        else:
            v = _git_small.get(key)
            if v is not None:
                return v
    out = run()
    if not out:
        return out
    with _cache_lock:
        if kind == "blame":
            if len(out) > _BLAME_MAX_BYTES:
                return out  # giant blames evict everything — skip caching
            if key in _blame_cache:
                _blame_cache.move_to_end(key)
            _blame_cache[key] = out
            while len(_blame_cache) > _BLAME_LRU_MAX:
                _blame_cache.popitem(last=False)
        else:
            _git_small[key] = out
    return out


# ── GitHub API pacing + per-repo lock ──
_api_min_gap = 0.0  # min seconds between API calls; 0 = unlimited (set via --api-qps)
_api_last_call = [0.0]
_api_pace_lock = threading.Lock()


def _pace_api() -> None:
    """Pace API calls to ~1 per _api_min_gap sec — keeps parallel per-repo processes
    inside the shared 5000/hr GitHub budget instead of 403-storming."""
    if _api_min_gap <= 0:
        return
    with _api_pace_lock:
        now = time.time()
        gap = _api_min_gap - (now - _api_last_call[0])
        if gap > 0:
            time.sleep(gap)
        _api_last_call[0] = time.time()


_API_TIMEOUT = 45  # sec/request — slow-but-successful (>15s) responses now succeed
_API_TRIES = 3     # retries on transient aborts/timeouts (each attempt is re-paced)


def _api_get(url: str, headers: dict, timeout: int = _API_TIMEOUT, tries: int = _API_TRIES):
    """Paced GET with retry — absorbs the intermittent resets/timeouts on this network.
    Same data on success, so it never changes verdicts; only fetch robustness."""
    import requests
    last = None
    for i in range(tries):
        _pace_api()
        try:
            return requests.get(url, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            if i < tries - 1:
                time.sleep(2.0 * (i + 1))  # 2s, 4s backoff
    raise last


def _pid_alive(pid: int) -> bool:
    """Process liveness check. os.kill(pid,0) is unreliable on Windows (reports dead
    PIDs as alive), so use the Win32 API directly."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return not ok or code.value == 259  # 259 == STILL_ACTIVE


def get_pr_meta(repo_full: str, pr_number: int) -> Dict:
    """Get PR merge SHA, merged_at, base SHA, and changed files with patches."""
    key = f"{repo_full}#{pr_number}"
    with _cache_lock:
        if key in _pr_meta_cache:
            return _pr_meta_cache[key]

    if _offline:
        print(f"  [offline] not in cache: {key}", file=sys.stderr)
        return {"merge_sha": "", "merged_at": "", "base_sha": "", "files": [], "body": "", "title": ""}

    import requests
    result = {"merge_sha": "", "merged_at": "", "base_sha": "", "files": [], "body": "", "title": ""}
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "User-Agent": "szz-tracer/1.0"}

    # PR details
    r = _api_get(f"https://api.github.com/repos/{repo_full}/pulls/{pr_number}", headers=headers)
    if r.status_code == 200:
        d = r.json()
        result["merge_sha"] = d.get("merge_commit_sha", "")
        result["merged_at"] = d.get("merged_at", "")
        result["base_sha"] = d.get("base", {}).get("sha", "")
        result["title"] = d.get("title", "")
        result["body"] = d.get("body", "") or ""

    # PR files (with patches)
    r2 = _api_get(f"https://api.github.com/repos/{repo_full}/pulls/{pr_number}/files?per_page=100",
                  headers={**headers, "Accept": "application/vnd.github.v3+json"})
    if r2.status_code == 200:
        for f in r2.json():
            patch = f.get("patch", "")
            # Extract line ranges from hunk headers
            ranges = re.findall(r'@@ -(\d+),?\d* \+(\d+),?\d* @@', patch)
            result["files"].append({
                "filename": f["filename"],
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "changes": f.get("changes", 0),
                "patch": patch,
                "pr_added_ranges": [(int(r[1]), int(r[1]) + 15) for r in ranges],
            })

    with _cache_lock:
        _pr_meta_cache[key] = result
    return result


# ── SZZ Core ──
# Note: AI author detection removed — Phase 0 already guarantees all PRs are AI-generated.
# SZZ verdicts rely purely on content-level diff overlap between AI PR additions and fix deletions.


def szz_blame(
    repo_dir: Path,
    fix_sha: str,
    filepath: str,
    pr_merge_sha: str,
    pr_added_ranges: List[Tuple[int, int]],
    pr_patch: str,
    base_sha: str = "",
) -> Optional[Dict]:
    """
    Full SZZ blame analysis on a single fix commit (blame-verified verdict).

    For every line deleted by the fix, `git blame -w -C` at fix^ identifies its
    introducing commit, and that commit is classified relative to the PR via git
    ancestry (PR-authored / PRE_PR / POST_MERGE). A deleted line counts as
    PR-related when:
      - its content matches PR-added text AND its introducer is PR-authored, or
        is post-merge (PR text preserved through a rewrite). A PRE_PR introducer
        with matching text is refuted as a content coincidence; or
      - its content does NOT match PR-added text but its introducer is
        PR-authored (refactor penetration: the PR wrote the line, its text
        drifted by fix time). Post-merge introducers get a depth-1 iterative
        re-blame to check whether the pre-rewrite line was PR-authored.

    Verdicts: SZZ_PASSED / LOW_BLAME_RATIO / PURE_ADDITIVE_NEAR|FAR /
    NO_BLAME_LINEAGE / NON_CODE_FILE. `base_sha` (the PR's base branch tip) is
    used instead of merge_sha^1 so rebase-and-merge PR commits are not misread
    as pre-PR; falls back to merge_sha^1 when unavailable.

    Introducer classification is merge-strategy aware: a squash/rebase merge
    (single parent) treats only the merge commit itself as PR-written and marks
    other base..merge commits CONCURRENT_PR (different PRs merged during this
    PR's lifetime — no longer misattributed); a merge-commit treats ancestors
    of merge^2 as the PR branch. See `_classify` for the full table.
    """
    # Get fix diff
    fix_diff = git_memo("diff", (str(repo_dir), f"{fix_sha}^..{fix_sha}", filepath),
                        lambda: run_git(repo_dir, ["diff", f"{fix_sha}^..{fix_sha}", "--", filepath]))
    if not fix_diff:
        return None

    additions, deletions = 0, 0
    added_lines: List[str] = []
    deleted_lines: List[str] = []
    for line in fix_diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
            added_lines.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
            deleted_lines.append(line[1:])

    if additions == 0 and deletions == 0:
        return None  # ghost commit — skip

    # git blame at fix^ (parent of fix) for the file
    blame_out = git_memo("blame", (str(repo_dir), f"{fix_sha}^", filepath),
                         lambda: run_git(repo_dir, ["blame", "-w", "-C", "-l", "--porcelain", f"{fix_sha}^", "--", filepath]))
    if not blame_out:
        # Fallback: blame at fix itself
        blame_out = git_memo("blame", (str(repo_dir), fix_sha, filepath),
                             lambda: run_git(repo_dir, ["blame", "-w", "-C", "-l", "--porcelain", fix_sha, "--", filepath]))

    # Parse blame output — track introducing commits for provenance
    # git blame --porcelain format per block:
    #   <sha> <orig> <final> <count>\n<header-fields>\n...\n\t<content>
    blame_entries: List[Dict] = []
    # Split on newline followed by a 40-hex-char SHA at line start
    blocks = re.split(r'\n(?=[0-9a-f]{40} \d+ \d+)', blame_out)
    for block in blocks:
        if not block.strip():
            continue
        # Separate header from content by the tab before the content line
        parts = block.split('\t', 1)
        header_lines = parts[0].strip().split('\n')
        content = parts[1].strip() if len(parts) > 1 else ''

        if not header_lines:
            continue
        header_parts = header_lines[0].split()
        if len(header_parts) < 2:
            continue
        sha = header_parts[0]

        author = ""
        for l in header_lines:
            if l.startswith("author "):
                author = l[7:]
                break

        blame_entries.append({
            "sha": sha,
            "author": author,
            "content": content,
        })

    # Build introducing commits index
    introducing_commits: Dict[str, Dict] = {}

    for be in blame_entries:
        sha = be["sha"]
        if sha not in introducing_commits:
            introducing_commits[sha] = {
                "sha": sha, "author": be["author"],
                "message": git_memo("msg", (str(repo_dir), sha),
                                    lambda: run_git(repo_dir, ["log", "-1", "--format=%s", sha])),
                "line_count": 0, "lines": [],
            }
        introducing_commits[sha]["line_count"] += 1
        introducing_commits[sha]["lines"].append({
            "line": len(introducing_commits[sha]["lines"]) + 1,
            "content": be["content"][:200],
        })

    # Overlap check: PR-added content vs fix-deleted content
    pr_added_content = set()
    for line in pr_patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            stripped = line[1:].strip()
            if len(stripped) > 5:
                pr_added_content.add(stripped)

    fix_deleted_content = set()
    for line in deleted_lines:
        stripped = line.strip()
        if len(stripped) > 5:
            fix_deleted_content.add(stripped)

    has_content_overlap = len(pr_added_content & fix_deleted_content) > 0

    # Proximity check for additive fixes (fix adds code near PR's code)
    proximity_overlap = False
    if additions > 0 and deletions <= 3:
        fix_added_ranges = []
        for m in re.finditer(r'@@ -\d+,?\d* \+(\d+),?\d* @@', fix_diff):
            start = int(m.group(1))
            fix_added_ranges.append((start, start + 15))

        PROXIMITY = 50
        for (pr_start, pr_end) in pr_added_ranges:
            for (fix_start, fix_end) in fix_added_ranges:
                if abs(fix_start - pr_start) <= PROXIMITY or abs(fix_end - pr_end) <= PROXIMITY:
                    proximity_overlap = True
                    break
            if proximity_overlap:
                break

    # ── Blame-verified verdict: classify each deleted line's introducer ──
    # Resolve the PR's base tip: prefer the API-provided base_sha (handles
    # rebase-and-merge correctly), fall back to the merge commit's first parent.
    base = base_sha
    if not base and pr_merge_sha:
        base = run_git(repo_dir, ["rev-parse", f"{pr_merge_sha}^1"])

    # Merge strategy: squash/rebase merges leave a single-parent merge commit,
    # so only the merge commit itself (ic == pr_merge_sha) is PR-written; a
    # merge-commit has ^2 = PR branch tip, so every ^2 ancestor is PR-written.
    _merge_second = None  # merge^2 sha, set only for merge-commit merges
    if pr_merge_sha:
        _parents = run_git(repo_dir, ["rev-list", "--parents", "-n", "1", pr_merge_sha])
        if _parents:
            _p = _parents.split()
            if len(_p) >= 3:
                _merge_second = _p[2]

    def _classify(ic: str) -> str:
        """PR / PRE_PR / POST_MERGE / CONCURRENT_PR / UNKNOWN for an introducing commit.

        "PR" strictly means *this PR* wrote the line:
          - ic == merge_sha (the squash/rebase-merge commit itself) -> PR
          - merge-commit (two parents): an ancestor of merge^2 that is NOT an
            ancestor of the PR's base is a PR-branch commit -> PR; anything on
            the merge^1 (mainline) side, or already present at base (old mainline
            that the branch point inherits), -> PRE_PR
          - single-parent (squash / rebase-and-merge): only merge_sha itself is
            PR-written. Any other ancestor is PRE_PR if it predates the PR's
            base, otherwise CONCURRENT_PR — a *different* PR merged during this
            PR's lifetime, whose commits land in base..merge and were wrongly
            attributed to this PR before the fix.
        """
        if not ic or not pr_merge_sha:
            return "UNKNOWN"
        if ic == pr_merge_sha:
            return "PR"  # the merge/squash commit itself
        anc_merge = is_ancestor(repo_dir, ic, pr_merge_sha)
        if anc_merge is None:
            return "UNKNOWN"
        if not anc_merge:
            return "POST_MERGE"  # written after the PR was merged
        # ic is an ancestor of the merge commit
        if _merge_second:  # merge-commit: PR branch == merge^2 side
            anc_pr = is_ancestor(repo_dir, ic, _merge_second)
            if anc_pr is None:
                return "UNKNOWN"
            if not anc_pr:
                return "PRE_PR"  # on the merge^1 (mainline) side
            # ic IS an ancestor of merge^2 — but so is every pre-base mainline
            # commit (through the branch point). Only post-base ^2 commits are
            # PR-written; anything already present at the PR's base is PRE_PR.
            anc_base = is_ancestor(repo_dir, ic, base) if base else None
            if anc_base is None:
                return "UNKNOWN"
            return "PRE_PR" if anc_base else "PR"
        # single-parent (squash / rebase-and-merge): non-merge commits in
        # base..merge belong to concurrent PRs, not this one
        anc_base = is_ancestor(repo_dir, ic, base) if base else None
        if anc_base is None:
            return "UNKNOWN"
        return "PRE_PR" if anc_base else "CONCURRENT_PR"

    def _reblame_prior_writer(ic: str, content: str) -> Optional[str]:
        """Depth-1 iterative re-blame. For a deleted line written by post-merge
        commit `ic`, find who wrote its pre-rewrite version: walk ic's diff, pair
        the added line == `content` with the removed line it replaced (by order
        within each hunk), then blame at ic^ for that pre-image. Returns None when
        the line was newly added by ic (no PR-origin chain to follow)."""
        diff = git_memo("diff", (str(repo_dir), f"{ic}^..{ic}", filepath),
                        lambda: run_git(repo_dir, ["diff", f"{ic}^..{ic}", "--", filepath]))
        if not diff:
            return None
        if len(diff) > _REBLAME_MAX_DIFF or diff.count("\n") > _REBLAME_MAX_DIFF_LINES:
            # Whole-file rewrite of a generated bundle: skip the re-blame rather
            # than re-parse a huge diff per deleted line. Conservative —
            # this line is not PR-attributable via re-blame, never a false +.
            return None

        def _pair_minus(minus: List[str], plus: List[str]) -> Optional[str]:
            for j, c in enumerate(plus):
                if c == content and j < len(minus):
                    return minus[j]
            return None

        minus, plus = [], []
        pre_image = None
        for ln in diff.split("\n"):
            if ln.startswith("@@"):
                if pre_image is None:
                    pre_image = _pair_minus(minus, plus)
                minus, plus = [], []
            elif ln.startswith("+") and not ln.startswith("+++"):
                plus.append(ln[1:].strip())
            elif ln.startswith("-") and not ln.startswith("---"):
                minus.append(ln[1:].strip())
        if pre_image is None:
            pre_image = _pair_minus(minus, plus)
        if pre_image is None:
            return None
        blame2 = git_memo("blame", (str(repo_dir), f"{ic}^", filepath),
                          lambda: run_git(repo_dir, ["blame", "-w", "-l", "--porcelain", f"{ic}^", "--", filepath]))
        if not blame2:
            return None
        for block in re.split(r'\n(?=[0-9a-f]{40} \d+ \d+)', blame2):
            if not block.strip():
                continue
            parts = block.split('\t', 1)
            if len(parts) < 2:
                continue
            if parts[1].strip() == pre_image:
                return parts[0].strip().split('\n')[0].split()[0]
        return None

    # deleted line content -> introducer sha (first match, full content not truncated)
    intro_by_content: Dict[str, str] = {}
    for be in blame_entries:
        c = (be.get("content") or "").strip()
        if c and c not in intro_by_content:
            intro_by_content[c] = be["sha"]

    deleted_line_evidence: List[Dict] = []
    for dl in sorted(fix_deleted_content):
        ic = intro_by_content.get(dl)
        pos = _classify(ic) if ic else "UNKNOWN"
        in_pr = dl in pr_added_content
        related, evidence = False, "unrelated"
        if in_pr:
            if pos == "PR":
                related, evidence = True, "content+blame-confirm"
            elif pos == "POST_MERGE":
                related, evidence = True, "content+post-merge-survival"
            elif pos == "PRE_PR":
                related, evidence = False, "coincidence-refuted"
            elif pos == "CONCURRENT_PR":
                related, evidence = False, "concurrent-pr-refuted"
            else:
                related, evidence = False, "unconfirmed"
        else:
            if pos == "PR":
                related, evidence = True, "refactor-penetration"
            elif pos == "POST_MERGE":
                prev = _reblame_prior_writer(ic, dl) if ic else None
                if prev and _classify(prev) == "PR":
                    related, evidence = True, "reblame-penetration"
                else:
                    related, evidence = False, "reblame-refuted"
            elif pos == "CONCURRENT_PR":
                related, evidence = False, "unrelated"
            else:
                related, evidence = False, "unrelated"
        deleted_line_evidence.append({
            "content": dl[:200],
            "introducer_sha": ic,
            "position_class": pos,
            "evidence": evidence,
            "related": related,
        })

    # Confidence — blame-verified related-line ratio (replaces pure content overlap)
    total_deleted = len(fix_deleted_content)
    overlap_count = len(pr_added_content & fix_deleted_content)
    related_count = sum(1 for e in deleted_line_evidence if e["related"])
    base_confidence = related_count / max(1, total_deleted) if total_deleted > 0 else 0
    pr_authored_introducers = sum(
        1 for _sha in {e["introducer_sha"] for e in deleted_line_evidence if e["introducer_sha"]}
        if _classify(_sha) == "PR"
    )

    is_pure_additive = deletions == 0 and additions > 0

    # Verdict — blame-confirmed lineage (min 3 related lines avoids single-line coincidences)
    if is_pure_additive:
        verdict = "PURE_ADDITIVE_NEAR" if proximity_overlap else "PURE_ADDITIVE_FAR"
    elif related_count >= 3 and base_confidence >= 0.2:
        verdict = "SZZ_PASSED"
    elif related_count > 0:
        verdict = "LOW_BLAME_RATIO"
    else:
        verdict = "NO_BLAME_LINEAGE"

    # Non-code penalty
    if is_non_code(filepath):
        base_confidence *= 0.1
        verdict = "NON_CODE_FILE"

    fix_msg = run_git(repo_dir, ["log", "-1", "--format=%s", fix_sha])

    return {
        "additions": additions,
        "deletions": deletions,
        "total_churn": additions + deletions,
        "code_lines_deleted": deletions,
        "code_lines_added": additions,
        "overlap_count": overlap_count,
        "truly_related_lines": related_count,
        "related_count": related_count,
        "pr_authored_introducers": pr_authored_introducers,
        "has_diff_overlap": has_content_overlap,
        "base_confidence": base_confidence,
        "ai_pr_modified_file": True,
        "is_pure_additive": is_pure_additive,
        "proximity_overlap": proximity_overlap,
        "fix_commit_message": fix_msg,
        "verdict": verdict,
        "introducing_commits": list(introducing_commits.values()),
        "_deleted_line_evidence": deleted_line_evidence,
        "_pr_added_content": list(pr_added_content)[:50],
        "_fix_deleted_content": list(fix_deleted_content)[:50],
    }


def track_downstream(
    repo_dir: Path,
    merge_sha: str,
    filepath: str,
    since_date: str,
) -> List[Dict]:
    """Find downstream commits that modified filepath after merge_sha."""
    log = git_memo("log", (str(repo_dir), f"{merge_sha}..HEAD", filepath),
                   lambda: run_git(
                       repo_dir,
                       ["log", f"{merge_sha}..HEAD", "--format=%H|%ad|%s", "--date=short", "--", filepath],
                   ))
    commits = []
    for line in log.split("\n"):
        parts = line.strip().split("|", 2)
        if len(parts) == 3:
            sha, date, msg = parts
            if date >= (since_date or "2024-01-01"):
                commits.append({"sha": sha, "date": date, "message": msg})
    return commits


def filter_comment_lines(diff_text: str) -> Tuple[int, int]:
    """Count comment lines in a diff for filtering stats."""
    comment_count = 0
    for line in diff_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("//", "#", "--", "/*", "*", "*/", "<!--", "-->", "'''", '"""')):
            comment_count += 1
        elif stripped.startswith(("+", "-")) and len(stripped) > 1:
            inner = stripped[1:].strip()
            if inner.startswith(("//", "#", "--", "/*", "*", "'''", '"""')):
                comment_count += 1
    return comment_count


# ── Main pipeline for one repo ──
def process_one_pr(
    repo_full: str,
    repo_dir: Path,
    pr: Dict,
    out_dir: Path,
    limit_per_pr: int = 0,
) -> List[Dict]:
    """Process a single AI PR through the full SZZ pipeline."""
    pr_number = int(pr["number"])
    case_prefix = f"{repo_full}#{pr_number}"

    # 1. Get PR metadata
    meta = get_pr_meta(repo_full, pr_number)
    merge_sha = meta["merge_sha"]
    merged_at = meta["merged_at"]
    pr_title = meta.get("title", "")
    pr_files = meta["files"]

    if not merge_sha or not pr_files:
        return []

    # Parse PR merge date for time decay calculation
    pr_merged_date = None
    if merged_at:
        try:
            pr_merged_date = datetime.fromisoformat(merged_at.replace("Z", "+00:00")).date()
        except (ValueError, TypeError):
            pass

    # 2. For each code file modified by the PR, find downstream fix commits
    results = []
    for pf in pr_files:
        fp = pf["filename"]
        if is_non_code(fp):
            continue  # skip non-code files

        pr_added_ranges = pf.get("pr_added_ranges", [])
        pr_patch = pf.get("patch", "")

        # Track downstream
        downstream = track_downstream(repo_dir, merge_sha, fp, merged_at[:10] if merged_at else "2024-01-01")
        if not downstream:
            continue

        # Limit per file
        if limit_per_pr > 0:
            downstream = downstream[:limit_per_pr]

        # 3. SZZ blame each downstream commit
        for dc in downstream:
            sha = dc["sha"]
            date = dc["date"]
            msg = dc["message"]

            blame_result = szz_blame(
                repo_dir, sha, fp, merge_sha, pr_added_ranges, pr_patch,
                base_sha=meta.get("base_sha", ""),
            )
            if not blame_result:
                continue

            # Build output entry
            intro_commits = blame_result.pop("introducing_commits", [])
            verdict = blame_result.pop("verdict", "UNKNOWN")

            # Comment filtering stats
            fix_diff = git_memo("diff", (str(repo_dir), f"{sha}^..{sha}", fp),
                                lambda: run_git(repo_dir, ["diff", f"{sha}^..{sha}", "--", fp]))
            comment_lines = filter_comment_lines(fix_diff) if fix_diff else 0

            # Calculate fix latency (time decay from PR merge to fix commit)
            fix_latency_days = None
            if pr_merged_date and date:
                try:
                    fix_date = datetime.strptime(date, "%Y-%m-%d").date()
                    fix_latency_days = (fix_date - pr_merged_date).days
                except (ValueError, TypeError):
                    pass

            entry = {
                "case_id": f"{case_prefix}#{fp.replace('/', '_')}#{sha[:8]}",
                "pr_number": pr_number,
                "commit_sha": sha,
                "filepath": fp,
                "commit_date": date,
                "commit_message": msg,
                "commit_author": "",
                "repo_name": repo_full,
                "agent": pr.get("agent", "unknown"),
                "cohort": pr.get("cohort", "AI-attributed"),
                "language": pr.get("_language", ""),
                "pr_merged_at": merged_at[:10] if merged_at else "",
                "pr_title": pr_title,
                "fix_latency_days": fix_latency_days,
                "verdict_level": verdict,
                "szz_traceability": {
                    "blame_summary": {
                        "total_code_lines_deleted": blame_result["deletions"],
                        "code_lines_added": blame_result["additions"],
                        "overlap_count": blame_result["overlap_count"],
                        "truly_related_lines": blame_result["truly_related_lines"],
                        "related_count": blame_result["related_count"],
                        "pr_authored_introducers": blame_result["pr_authored_introducers"],
                        "base_confidence": blame_result["base_confidence"],
                        "ai_pr_modified_file": blame_result["ai_pr_modified_file"],
                        "has_diff_overlap": blame_result["has_diff_overlap"],
                        "is_pure_additive": blame_result["is_pure_additive"],
                        "fix_commit_message": blame_result["fix_commit_message"],
                    },
                    "introducing_commits": intro_commits,
                    "_deleted_line_evidence": blame_result["_deleted_line_evidence"],
                    "filtering_stats": {
                        "code_lines_added": blame_result["additions"],
                        "code_lines_deleted": blame_result["deletions"],
                        "comment_lines_filtered": comment_lines,
                    },
                    "confidence": blame_result["base_confidence"],
                },
                "is_td": None,
                "td_type": None,
                "boundary_rule": "",
                "is_related": None,
            }
            results.append(entry)

    return results


def run_szz_pipeline(
    repo_full: str,
    limit_prs: int = 0,
    limit_per_pr: int = 0,
    workers: int = 10,
    resume: bool = True,
    prs_file: Optional[Path] = None,
    output_root: Optional[Path] = None,
    cohort_label: str = "AI-attributed",
) -> List[Dict]:
    """Run full SZZ pipeline for one repo.

    Optional manifest/output parameters allow the same tracer to be reused
    for a matched control cohort while preserving the original AI defaults.
    """
    repo_name = repo_full.split("/")[1]
    repo_dir = REPO_BASE / repo_full.split("/")[1]

    if not repo_dir.exists() or not (repo_dir / ".git").exists():
        raise FileNotFoundError(f"Repo not found at {repo_dir}")

    # Single-instance per-repo guard (parallel runs: one process per repo).
    # A stale lock from a dead process is auto-overridden via _pid_alive.
    lock_file = repo_dir / ".szz.lock"
    my_pid = os.getpid()
    if lock_file.exists():
        try:
            other = int(lock_file.read_text().strip() or "0")
            if other != my_pid and _pid_alive(other):
                print(f"  [lock] {repo_full} already being processed by PID {other} — skipping")
                return []
        except ValueError:
            pass  # unreadable/stale lock — take over below
    lock_file.write_text(str(my_pid))
    print(f"  [lock] acquired PID {my_pid}")

    def _release_lock():
        if lock_file.exists():
            try:
                lock_file.unlink()
            except OSError:
                pass

    # Ensure output dir
    output_root = output_root or (SRC_DIR / "output")
    out_dir = output_root / repo_full.split("/")[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "02_szz_traceability.json"

    # Load PI PRs for this repo
    with open(prs_file or FINAL_PRS, "r", encoding="utf-8") as f:
        all_prs = json.load(f)
    prs = [r for r in all_prs if r["repo"] == repo_full]
    print(f"Repo: {repo_full} | {cohort_label} PRs: {len(prs)} | Dir: {repo_dir}")

    if limit_prs > 0:
        prs = prs[:limit_prs]

    # Resume from existing
    results = []
    processed_ids = set()
    if resume and out_file.exists():
        if out_file.stat().st_size > 500 * 1024 * 1024:
            # Large 02 (bun/gumroad/fiber/remotion are 3-6 GB): json.load of the
            # whole file spikes memory (raw string + objects), risking OOM next to
            # the results list. Stream top-level items with ijson instead.
            import ijson  # required for streaming large SZZ files
            with open(out_file, "rb") as f:
                for item in ijson.items(f, "item", use_float=True):
                    results.append(item)
        else:
            with open(out_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        processed_ids = {r["case_id"] for r in results}
        print(f"  Resuming from {len(results)} existing cases")

    todo = [pr for pr in prs if int(pr["number"]) not in {r["pr_number"] for r in results}]
    print(f"  To process: {len(todo)} PRs ({len(results)} cached)")

    if not todo:
        _release_lock()
        return results

    save_lock = threading.Lock()
    stats = {"processed": 0, "new_cases": 0}
    last_save_count = len(results)

    def _save():
        nonlocal last_save_count
        with save_lock:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_save_count = len(results)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_one_pr, repo_full, repo_dir, pr, out_dir, limit_per_pr): pr
            for pr in todo
        }
        pbar = tqdm(total=len(todo), desc="SZZ", unit="pr") if use_tqdm else None

        for f in as_completed(futures):
            pr = futures[f]
            try:
                pr_results = f.result()
                if pr_results:
                    results.extend(pr_results)
                    stats["new_cases"] += len(pr_results)
                stats["processed"] += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"cases": len(results), "new": stats["new_cases"]})

                # Incremental save: every 50 new cases or at end of each PR
                if len(results) - last_save_count >= 50:
                    _save()
            except Exception as e:
                stats["processed"] += 1
                if pbar:
                    pbar.update(1)
                print(f"\n  ERROR PR#{pr.get('number', '?')}: {e}")

        if pbar:
            pbar.close()

    _save()

    # Summary
    passed = sum(1 for r in results if r["verdict_level"] == "SZZ_PASSED")
    non_code = sum(1 for r in results if r["verdict_level"] == "NON_CODE_FILE")
    print(f"\n  Done: {len(results)} candidates | SZZ_PASSED={passed} | NON_CODE={non_code}")
    print(f"  Saved to: {out_file}")
    _release_lock()
    return results


# ── CLI ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SZZ Tracer — TD lineage tracing")
    parser.add_argument("--repo", required=True, help="e.g. microsoft/aspire")
    parser.add_argument("--limit", type=int, default=0, help="Limit AI PRs to process")
    parser.add_argument("--limit-per-pr", type=int, default=0, help="Max downstream commits per PR file")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--api-qps", type=float, default=0.0,
                        help="Max GitHub API calls/sec for this process (0=unlimited)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    parser.add_argument("--offline", action="store_true",
                        help="Never call GitHub API; rely on pr_meta_cache.json (prefilled by prefetch_pr_meta.py)")
    parser.add_argument("--prs-file", type=Path, default=None,
                        help="Alternative PR manifest, such as matched controls")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Output root; defaults to src/output")
    parser.add_argument("--cohort-label", default="AI-attributed")
    args = parser.parse_args()

    if args.api_qps > 0:
        _api_min_gap = 1.0 / args.api_qps
    if args.offline:
        _offline = True

    t0 = time.time()
    run_szz_pipeline(
        repo_full=args.repo,
        limit_prs=args.limit,
        limit_per_pr=args.limit_per_pr,
        workers=args.workers,
        resume=not args.no_resume,
        prs_file=args.prs_file,
        output_root=args.output_root,
        cohort_label=args.cohort_label,
    )
    print(f"Total time: {time.time() - t0:.0f}s")

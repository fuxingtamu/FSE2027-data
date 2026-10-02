"""Build progressive L1-L5 repository contexts for RQ3 cases.

The levels range from the PR diff only (L1) to surrounding function, full file,
other files in the same PR, and repository conventions (L5).
"""
import json, re, sys, io, subprocess, urllib.request, urllib.error
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rq3_config import (MERGED_FILE, MANIFEST_FILE, CONTEXT_FILE,
                          COMMITS_CACHE, REPOS_DIR, ENV_FILE)

# Context window and truncation limits.
L2_WINDOW = 40
MAX_FILE_LINES = 400
MAX_SIBLING_FILES = 4
MAX_SIBLING_LINES = 200
MAX_CONV_FILES = 3
MAX_CONV_LINES = 200

CONVENTION_PATHS = [
    "README.md", "README.rst", "CONTRIBUTING.md", "CONTRIBUTORS.md",
    "CODE_OF_CONDUCT.md", "ARCHITECTURE.md", "docs/ARCHITECTURE.md",
    "pyproject.toml", "setup.py", "setup.cfg", "package.json",
    "go.mod", "Cargo.toml", ".eslintrc.js", ".eslintrc.json", "tsconfig.json",
]


def load_env_token() -> str:
    try:
        for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            body = line[len('export '):].strip() if line.startswith('export ') else line
            if body.startswith('GITHUB_TOKEN'):
                return body.split('=', 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def github_api(repo, pr, token):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr}"
    import requests
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "rq3v2-context-extract"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    d = response.json()
    return {
        'merge_commit_sha': d.get('merge_commit_sha'),
        'base_sha': d.get('base', {}).get('sha'),
        'head_sha': d.get('head', {}).get('sha'),
        'title': d.get('title'),
        'state': d.get('state'),
    }


def git(args, repo_dir):
    return subprocess.run(['git', '-C', str(repo_dir)] + args,
                          capture_output=True, text=True, encoding='utf-8', errors='replace')


def find_merge_git(repo_dir, pr, title):
    r = git(['log', '--all', '--format=%H%x09%P%x09%s', f'--grep=#{pr}'], repo_dir)
    cands = [l.split('\t') for l in r.stdout.splitlines() if l.strip() and '\t' in l]
    scored = []
    for c in cands:
        if len(c) < 3:
            continue
        sha, parents, subj = c[0], c[1], c[2]
        s = 0
        if title and title in subj:
            s += 4
        if re.search(rf'Merge pull request #{pr}\b', subj):
            s += 3
        if re.search(rf'\(#{pr}\)', subj):
            s += 2
        if re.search(rf'#{pr}\b', subj):
            s += 1
        if len(parents.split()) >= 2:
            s += 1
        scored.append((s, sha, len(parents.split()), subj))
    scored.sort(key=lambda x: -x[0])
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return None


def resolve_merge_commit(repo, pr, title, repo_dir, token, cache):
    key = f"{repo}#{pr}"
    if key in cache and cache[key].get('merge_commit_sha'):
        return cache[key]['merge_commit_sha']
    if token:
        try:
            info = github_api(repo, pr, token)
            cache[key] = info
            if info.get('merge_commit_sha'):
                return info['merge_commit_sha']
        except Exception as e:
            print(f"    [API fail {repo}#{pr}: {e}]")
    sha = find_merge_git(repo_dir, pr, title)
    if sha:
        cache[key] = {'merge_commit_sha': sha, 'base_sha': None, 'head_sha': None,
                      'title': title, 'state': 'closed', 'source': 'git'}
        return sha
    cache[key] = {'merge_commit_sha': None, 'source': 'none'}
    return None


def resolve_pr_snapshot(repo, pr, title, repo_dir, token, cache):
    """Resolve the PR-time code snapshot and retain its provenance.

    The review context must come from the PR head, not from the post-merge
    tree.  Older caches may contain only a merge SHA, so we refresh metadata
    from GitHub when possible and fall back to the PR side of a merge commit
    for legacy records.
    """
    key = f"{repo}#{pr}"
    info = dict(cache.get(key) or {})
    merge = info.get("merge_commit_sha") or info.get("merge_sha")
    if not merge:
        # Local history is deterministic and avoids a network dependency when
        # the repository clone already contains the merged PR.
        merge = find_merge_git(repo_dir, pr, title)
        if merge:
            info["merge_commit_sha"] = merge
    if not merge:
        merge = resolve_merge_commit(repo, pr, title, repo_dir, token, cache)
    # Prefer local commit topology.  This avoids an unnecessary API request
    # for ordinary two-parent merges and makes the provenance auditable.
    head = info.get("head_sha")
    source = "github_head" if head else "unresolved"
    if not head and merge:
        r = git(["rev-list", "--parents", "-n", "1", merge], repo_dir)
        parts = r.stdout.strip().split()
        if len(parts) >= 3:
            head = parts[2]
            source = "merge_second_parent_local"
        elif len(parts) == 2:
            # A one-parent SHA is usually a squash/rebase result. Its tree may
            # include base-branch changes, so obtain the original PR head.
            if token:
                try:
                    fresh = github_api(repo, pr, token)
                    info.update({k: v for k, v in fresh.items() if v})
                    head = info.get("head_sha")
                    merge = info.get("merge_commit_sha") or merge
                    source = "github_head" if head else "unresolved"
                except Exception as e:
                    print(f"    [single-parent snapshot API fail {repo}#{pr}: {e}]")
            if not head:
                head = merge
                source = "single_parent_merge_fallback"

    # For legacy cached merge commits, use the PR parent of a normal merge
    # commit.  This is only a fallback; squash/rebase records should be
    # refreshed from GitHub before the final replay.
    if not head and token:
        try:
            fresh = github_api(repo, pr, token)
            info.update({k: v for k, v in fresh.items() if v})
            head = info.get("head_sha")
            merge = info.get("merge_commit_sha") or merge
            source = "github_head" if head else "unresolved"
        except Exception as e:
            print(f"    [snapshot API fail {repo}#{pr}: {e}]")

    info.update({"merge_commit_sha": merge, "head_sha": head,
                 "snapshot_sha": head, "snapshot_source": source})
    cache[key] = info
    return info


def show_file(repo_dir, commit, filepath):
    r = git(['show', f'{commit}:{filepath}'], repo_dir)
    return r.stdout if r.returncode == 0 else None


def with_line_numbers(text, start=1):
    lines = text.splitlines()
    w = len(str(start + len(lines)))
    return "\n".join(f"{start + i:>{w}}  {ln}" for i, ln in enumerate(lines))


def changed_new_ranges(patch_text):
    ranges = []
    for m in re.finditer(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', patch_text):
        new_start = int(m.group(3))
        new_cnt = int(m.group(4) or 1)
        if new_cnt > 0:
            ranges.append((new_start, new_start + new_cnt - 1))
    return ranges


def merge_windows(ranges, win, n_lines):
    if not ranges:
        return []
    windows = []
    for s, e in ranges:
        windows.append([max(1, s - win), min(n_lines, e + win)])
    windows.sort()
    merged = [windows[0]]
    for w in windows[1:]:
        if w[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], w[1])
        else:
            merged.append(w)
    return merged


def render_window(text_lines, windows):
    parts = []
    for s, e in windows:
        block = text_lines[s - 1:e]
        parts.append(f"[lines {s}-{e}]\n" + with_line_numbers("\n".join(block), start=s))
    return "\n\n".join(parts)


def truncate_lines(text, max_lines, note_path=""):
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, len(lines)
    kept = "\n".join(lines[:max_lines])
    return (f"{kept}\n... [truncated: {len(lines)} lines total, showing first {max_lines}]",
            len(lines))


def main():
    data = json.load(open(MERGED_FILE, encoding='utf-8'))
    td = [r for r in data if r.get('_llm_is_td') is True]
    # Keep only cases that have a source patch.
    td_has_patch = [r for r in td if (r.get('pr_file_patch') or '').strip()]
    print(f"TD cases: {len(td)}; cases with patches: {len(td_has_patch)}")

    token = load_env_token()
    cache = {}
    if COMMITS_CACHE.exists():
        cache = json.load(open(COMMITS_CACHE, encoding='utf-8'))

    # Resolve each unique PR only once.
    seen = {}
    for r in td_has_patch:
        key = f"{r['repo_name']}#{r['pr_number']}"
        seen.setdefault(key, r)
    print(f"Unique PRs: {len(seen)}; resolving merge commits...")
    commit_map, repo_dir_map = {}, {}
    for i, (key, r) in enumerate(seen.items()):
        repo_dir = REPOS_DIR / r['repo_name'].split('/')[-1]
        sha = resolve_merge_commit(r['repo_name'], r['pr_number'], r.get('pr_title', ''),
                                   repo_dir, token, cache)
        commit_map[key] = sha
        repo_dir_map[key] = str(repo_dir)
        if (i + 1) % 20 == 0:
            print(f"  Resolved {i+1}/{len(seen)}")
    COMMITS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMITS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Keep output indices aligned with the source case list.
    td_idx = {id(r): i for i, r in enumerate(td)}
    out = []
    n_missing_merge = n_missing_file = 0
    for r in td_has_patch:
        idx = td_idx[id(r)]
        key = f"{r['repo_name']}#{r['pr_number']}"
        repo_dir = Path(repo_dir_map[key])
        filepath = r['filepath']
        merge = commit_map.get(key)
        patch = r.get('pr_file_patch', '') or ''
        entry = {
            'idx': idx, 'repo': r['repo_name'], 'pr_number': r['pr_number'],
            'filepath': filepath, 'case_id': r.get('case_id', ''),
            'td_category': r.get('_llm_td_category', '?'),
            'merge_commit_sha': merge,
        }

        post_file = show_file(repo_dir, merge, filepath) if merge else None
        if post_file is None and merge:
            base_file = show_file(repo_dir, f'{merge}^', filepath)
            post_file = base_file
            entry['file_src'] = 'base(parent)'
        elif post_file is not None:
            entry['file_src'] = 'merge'

        post_lines = post_file.splitlines() if post_file else []
        n_lines = len(post_lines)
        ranges = changed_new_ranges(patch)

        l1 = ""
        if post_file and ranges:
            windows = merge_windows(ranges, L2_WINDOW, n_lines)
            body = render_window(post_lines, windows)
            l2 = (f"## Surrounding code (function/block context) for `{filepath}`\n"
                  f"(window of ±{L2_WINDOW} lines around each changed hunk)\n\n{body}")
        else:
            l2 = ""
        if post_file:
            txt, total = truncate_lines(post_file, MAX_FILE_LINES)
            l3 = f"## Full changed file: `{filepath}` ({total} lines)\n\n{with_line_numbers(txt)}"
        else:
            l3 = ""

        siblings = []
        for s in sorted(r.get('pr_files_summary', []),
                        key=lambda x: -(x.get('additions', 0) + x.get('deletions', 0))):
            fn = s.get('filename', '')
            if not fn or fn == filepath:
                continue
            if len(siblings) >= MAX_SIBLING_FILES:
                break
            stxt = show_file(repo_dir, merge, fn) if merge else None
            if stxt:
                tt, total = truncate_lines(stxt, MAX_SIBLING_LINES)
                siblings.append(f"### `{fn}` ({total} lines)\n{with_line_numbers(tt)}")
        l4 = l3
        if siblings:
            l4 = l3 + "\n\n## Other files changed in this PR\n" + "\n\n".join(siblings)

        convs = []
        if merge:
            for cp in CONVENTION_PATHS:
                if len(convs) >= MAX_CONV_FILES:
                    break
                ctxt = show_file(repo_dir, merge, cp)
                if ctxt:
                    tt, total = truncate_lines(ctxt, MAX_CONV_LINES)
                    convs.append(f"### `{cp}` ({total} lines)\n{with_line_numbers(tt)}")
        l5 = l4
        if convs:
            l5 = l4 + "\n\n## Repository conventions / documentation\n" + "\n\n".join(convs)

        entry['levels'] = {'L1': l1, 'L2': l2, 'L3': l3, 'L4': l4, 'L5': l5}
        entry['n_lines'] = n_lines
        entry['changed_ranges'] = ranges
        entry['n_siblings'] = len(siblings)
        entry['n_conventions'] = len(convs)

        if not merge:
            n_missing_merge += 1
        if not post_file:
            n_missing_file += 1
        out.append(entry)

    CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTEXT_FILE, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    from collections import Counter
    lv = Counter()
    for e in out:
        for k in ('L1', 'L2', 'L3', 'L4', 'L5'):
            if e['levels'][k]:
                lv[k] += 1
    print("\n" + "=" * 70)
    print(f"Context extraction complete: {len(out)} cases -> {CONTEXT_FILE}")
    print(f"  Missing merge commits: {n_missing_merge}; missing changed files: {n_missing_file}")
    print(f"  Cases with non-empty context by level: {dict(lv)}")


if __name__ == '__main__':
    main()

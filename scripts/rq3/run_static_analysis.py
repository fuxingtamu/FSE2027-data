"""
RQ3 static-analysis replay for repair-linked CTD members.

When a case supplies ``static_snapshot_sha``, analyzers run on the original
PR-head snapshot before merge. Otherwise, the legacy parent-of-repair revision
is used. For pre-merge runs, lines changed by the later repair are mapped back
to identical lines in the PR snapshot before matching analyzer diagnostics.
"""
import json, os, sys, io, subprocess, tempfile, re, shutil, difflib
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

from rq3_config import MERGED_FILE, REPOS_DIR, STATIC_FILE, USE_FULL_SCALE, MANIFEST_FILE

GOLANGCI = os.environ.get("GOLANGCI_LINT", "golangci-lint")
if not shutil.which(GOLANGCI):
    # winget installs the binary under a per-user package directory that is
    # not always added to PATH in child processes on Windows.
    winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(winget_root.glob("GolangCI.golangci-lint_*/*/golangci-lint.exe"))
    if candidates:
        GOLANGCI = str(candidates[-1])


def run_proc(cmd, timeout=60, text=True, **kw):
    """Run a subprocess with a timeout that also handles child processes.

    On Windows, communicate(timeout=...) may hang while joining reader threads
    if grandchildren keep stdout or stderr pipes open. Wait on the main process,
    use daemon reader threads, and terminate the full process tree on timeout.
    """
    import threading as _th
    kw.setdefault('stdout', subprocess.PIPE)
    kw.setdefault('stderr', subprocess.PIPE)
    if os.name == 'nt':
        kw['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(cmd, **kw)
    except Exception:
        return None, '', ''
    out_buf, err_buf = [], []

    def pump(stream, buf):
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                buf.append(chunk)
        except Exception:
            pass

    t1 = _th.Thread(target=pump, args=(proc.stdout, out_buf), daemon=True)
    t2 = _th.Thread(target=pump, args=(proc.stderr, err_buf), daemon=True)
    t1.start()
    t2.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            try:
                subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)],
                               capture_output=True, timeout=15)
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        raise
    t1.join(timeout=10)
    t2.join(timeout=10)
    out = b''.join(out_buf).decode('utf-8', errors='replace')
    err = b''.join(err_buf).decode('utf-8', errors='replace')
    return code, out, err

LANG_LINTERS = {
    '.py': ['ruff', 'mypy'],
    '.ts': ['oxlint'],
    '.tsx': ['oxlint'],
    '.mjs': ['oxlint'],
    '.js': ['oxlint'],
    '.vue': ['oxlint'],
    '.go': ['golangci-lint'],
}

GOLANGCI_ENABLED = ("govet,ineffassign,errcheck,staticcheck,unused,misspell,"
                    "unconvert,whitespace,errorlint")
GOLANGCI_ENABLED_EXTENDED = (GOLANGCI_ENABLED +
                    ",gocritic,revive,nilerr,nilnesserr,nilnil,rowserrcheck,"
                    "bodyclose,sqlclosecheck,forcetypeassert,errchkjson,gosec,wrapcheck")


def get_file_at_commit(repo_path, commit_sha, filepath):
    try:
        code, out, err = run_proc(
            ["git", "show", f"{commit_sha}:{filepath}"], 30, cwd=repo_path)
    except Exception:
        return None, 'timeout'
    if code != 0:
        return None, (err or '').strip()
    return out, None


def get_fix_diff(repo_path, fix_sha, filepath):
    try:
        code, out, err = run_proc(
            ["git", "show", "-p", fix_sha, "--", filepath], 30, cwd=repo_path)
    except Exception:
        return None
    return out if code == 0 else None


def parse_fix_changed_lines(fix_diff):
    """Parse a unified diff and return changed or deleted old-file line numbers."""
    changed = set()
    cur = 0
    for line in (fix_diff or '').split('\n'):
        if line.startswith('@@'):
            m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if m:
                cur = int(m.group(1)) - 1
        elif line.startswith('-') and not line.startswith('---'):
            cur += 1
            changed.add(cur)
        elif line.startswith('+') and not line.startswith('+++'):
            pass
        elif not line.startswith('@@'):
            cur += 1
    return changed


def map_changed_lines_to_snapshot(snapshot_text, before_fix_text, changed_before_fix):
    """Map pre-fix line numbers back to the original PR snapshot via unchanged lines."""
    if snapshot_text is None or before_fix_text is None:
        return set()
    snapshot_lines = snapshot_text.splitlines()
    before_fix_lines = before_fix_text.splitlines()
    matcher = difflib.SequenceMatcher(a=snapshot_lines, b=before_fix_lines)
    reverse = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            reverse[block.b + offset + 1] = block.a + offset + 1
    return {reverse[line] for line in changed_before_fix if line in reverse}


def run_ruff(content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(content)
        tmp = f.name
    try:
        code, out, err = run_proc(["ruff", "check", tmp, "--output-format=json", "--no-cache",
                                   "--select=ALL", "--preview"], 60)
        raw = (out or '').strip()
        if not raw:
            return {}
        data = json.loads(raw)
    except Exception:
        return {}
    finally:
        os.unlink(tmp)
    out = defaultdict(list)
    for item in data:
        row = item.get("location", {}).get("row", 0)
        out[row].append(f"{item.get('code', '?')}: {item.get('message', '')}")
    return dict(out)


def run_mypy(content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(content)
        tmp = f.name
    try:
        code, out, err = run_proc(["mypy", tmp, "--ignore-missing-imports", "--no-error-summary",
                                   "--show-error-codes", "--no-color-output"], 60)
        output = (out or '').strip()
    except Exception:
        output = ""
    finally:
        os.unlink(tmp)
    out = defaultdict(list)
    for line in output.split('\n'):
        m = re.match(r'.*?:(\d+):\s*(?:error|warning):\s*(.+?)\s*\[(.+?)\]', line)
        if m:
            out[int(m.group(1))].append(f"{m.group(3)}: {m.group(2)}")
    return dict(out)


def run_oxlint(content, ext):
    suffix = ext if ext.startswith('.') else f'.{ext}'
    is_tsx = suffix in ('.tsx', '.jsx')
    is_vue = suffix == '.vue'
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        f.write(content)
        tmp = f.name
    try:
        cmd = ["oxlint", "-D", "all", "--import-plugin", "--jest-plugin", "--format=json"]
        if is_tsx:
            cmd.append("--react-plugin")
        if is_vue:
            cmd.append("--vue-plugin")
        cmd.append(tmp)
        # On Windows, invoke the oxlint .cmd shim through cmd.exe.
        if os.name == 'nt':
            cmd = ["cmd", "/c"] + cmd
        code, out, err = run_proc(cmd, 60)
        raw = (out or '').strip()
        if not raw:
            return {}
        data = json.loads(raw)
    except Exception:
        return {}
    finally:
        os.unlink(tmp)
    out = defaultdict(list)
    diag = data.get("diagnostics", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for item in diag:
        labels = item.get("labels", [])
        if labels and isinstance(labels[0], dict):
            ln = labels[0].get("span", {}).get("line", 0)
        else:
            ln = item.get("startLine", item.get("line", 0))
        rule = item.get("code", item.get("ruleId", "?"))
        if ln:
            out[ln].append(f"{rule}: {item.get('message', '')}")
    return dict(out)


def run_golangci_worktree(repo_path, ai_sha, filepath):
    """Run golangci-lint in a temporary worktree and return target-file findings.

    A temporary single-file module cannot resolve imports reliably, so analysis
    runs against the repository tree at the selected AI snapshot.
    """
    import tempfile as tf
    issues = defaultdict(list)
    wt = td = None
    try:
        td = tf.mkdtemp(prefix='rq3wt_')
        wt = Path(td) / 'wt'
        code, out, err = run_proc(['git', 'worktree', 'add', '--detach', str(wt), ai_sha],
                                  180, cwd=repo_path)
        if code != 0:
            return {'__error__': f'worktree creation failed: {(err or "").strip()[:80]}'}

        rel = Path(filepath)
        cur = rel.parent
        while cur != Path('.'):
            if (wt / str(cur) / 'go.mod').exists():
                break
            cur = cur.parent
        moddir = wt / str(cur)
        pkgrel = rel.relative_to(cur).parent  # Package path relative to the Go module.
        if (wt / filepath).read_text(encoding='utf-8', errors='replace').find('import "C"') >= 0:
            return {'__error__': 'Cannot build this cgo package because gcc is unavailable.'}

        code2, out2, err2 = run_proc(
            [GOLANGCI, 'run', '--no-config', '--default=none',
             f'--enable={GOLANGCI_ENABLED_EXTENDED if os.environ.get("RQ3_GO_EXTENDED") == "1" else GOLANGCI_ENABLED}',
             '--output.json.path', 'stdout',
             '--', f'./{pkgrel.as_posix()}' if str(pkgrel) != '.' else './'],
            120, cwd=moddir)
        raw = (out2 or '').strip()
        err2 = err2 or ''
        if 'no go files to analyze' in err2 or 'context loading failed' in err2:
            return {'__error__': err2.strip()[:80]}
        if not raw:
            return dict(issues)
        start, end = raw.find('{'), raw.rfind('}')
        data = json.loads(raw[start:end + 1]) if end > start else {}
        target = rel.as_posix()
        for it in data.get('Issues', []):
            fn = (it.get('Pos', {}) or {}).get('Filename', '')
            if fn.replace('\\', '/') == target:
                line = (it.get('Pos', {}) or {}).get('Line', 0)
                if line:
                    issues[line].append(f"{it.get('FromLinter', '?')}: {it.get('Text', '')}")
    except Exception as e:
        return {'__error__': str(e)[:80]}
    finally:
        if wt is not None:
            try:
                run_proc(['git', 'worktree', 'remove', '--force', str(wt)], 180, cwd=repo_path)
            except Exception:
                pass
    return dict(issues)


def run_linter(content, ext):
    langs = LANG_LINTERS.get(ext.lower())
    if not langs:
        return None, None
    per, all_lines = {}, set()
    for lang in langs:
        if lang == 'ruff':
            v = run_ruff(content)
        elif lang == 'mypy':
            v = run_mypy(content)
        elif lang == 'oxlint':
            v = run_oxlint(content, ext)
        else:
            v = {}
        per[lang] = v
        all_lines |= set(k for k in v.keys() if k != '__error__')
    return per, all_lines


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='', help='Comma-separated manifest indices (default: all)')
    ap.add_argument('--out', default='', help='Output JSON path (default: configured static result path)')
    args = ap.parse_args()
    case_filter = set(int(x) for x in args.cases.split(',') if x.strip()) if args.cases else None
    out_file = Path(args.out) if args.out else STATIC_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)

    input_file = Path(os.environ.get('RQ3_MERGED_FILE', str(MERGED_FILE)))
    data = json.load(open(input_file, encoding='utf-8'))
    td = [r for r in data if r.get('_llm_is_td') is True]
    if USE_FULL_SCALE:
        # The latest obligation-aware population is authoritative for the
        # full-scale run.  The classifier file still contains the older 3,006
        # file-level records, including records excluded from the obligation
        # table and members without a usable patch.
        manifest = json.load(open(MANIFEST_FILE, encoding='utf-8'))
        keep_ids = {m['case_id'] for m in manifest if m.get('diff_file')}
        td = [r for r in td if r.get('case_id') in keep_ids]
    if case_filter is not None:
# Map selected manifest indices to case IDs.
        manifest = json.load(open(MANIFEST_FILE, encoding='utf-8'))
        keep_ids = {m['case_id'] for m in manifest if m['idx'] in case_filter}
        td = [r for r in td if r.get('case_id') in keep_ids]
    print(f"TD cases selected: {len(td)}")

    results = []
    n_ok = n_skip_lang = n_skip_file = n_skip_go = n_err = 0

    idx_of = {m['case_id']: m['idx'] for m in json.load(open(MANIFEST_FILE, encoding='utf-8'))}

    for case in td:
        cid = case.get('case_id', '')
        repo = (case.get('repo_name') or '').split('/')[-1]
        pr = str(case.get('pr_number'))
        filepath = case.get('filepath', '')
        fix_sha = case.get('commit_sha', '')
        cat = case.get('_llm_td_category', '?')
        ext = Path(filepath).suffix.lower()

        rec = {'case_id': case.get('case_id', ''), 'pr_number': pr, 'repo': repo,
               'filepath': filepath, 'language': ext, 'td_category': cat,
               'fix_sha': fix_sha[:10], 'status': '?'}

        if ext not in LANG_LINTERS:
            rec['status'] = 'skip_no_linter'
            rec['note'] = f'No configured linter for file extension {ext}'
            results.append(rec)
            n_skip_lang += 1
            continue

        repo_path = REPOS_DIR / repo
        if not repo_path.exists():
            rec['status'] = 'error'
            rec['note'] = f'Repository directory does not exist: {repo_path}'
            results.append(rec)
            n_err += 1
            continue

        # For the PR-head replay, the input builder supplies the head snapshot.
        # Without it, preserve the legacy parent-of-fix behavior.
        ai_sha = case.get('static_snapshot_sha') or f"{fix_sha}^"
        rec['ai_sha'] = ai_sha[:10]
        rec['snapshot_sha'] = ai_sha

        if ext == '.go':
            # Go analysis requires the repository module context.
            per = {'golangci-lint': run_golangci_worktree(repo_path, ai_sha, filepath)}
            all_lines = set(k for k in per['golangci-lint'].keys() if k != '__error__')
            if '__error__' in per['golangci-lint']:
                rec['status'] = 'skip_golangci'
                rec['note'] = f"golangci-lint could not analyze this case: {per['golangci-lint']['__error__']}"
                results.append(rec)
                n_skip_go += 1
                continue
            rec['golangci_worktree'] = True
        else:
            content, err = get_file_at_commit(repo_path, ai_sha, filepath)
            if content is None:
                rec['status'] = 'skip_file_missing'
                rec['note'] = f'{filepath} is unavailable at fix parent {ai_sha[:8]}: {err[:100]}'
                results.append(rec)
                n_skip_file += 1
                continue
            per, all_lines = run_linter(content, ext)
        rec['linter_lines'] = {k: len(v) for k, v in per.items() if k != '__error__'}
        rec['linter_details'] = per

        fix_diff = get_fix_diff(repo_path, fix_sha, filepath)
        fix_changed = parse_fix_changed_lines(fix_diff) if fix_diff else set()
        if case.get('static_snapshot_sha'):
            snapshot_text = content if ext != '.go' else get_file_at_commit(repo_path, ai_sha, filepath)[0]
            pre_fix_text = get_file_at_commit(repo_path, f"{fix_sha}^", filepath)[0]
            candidate_lines = map_changed_lines_to_snapshot(snapshot_text, pre_fix_text, fix_changed)
        else:
            candidate_lines = fix_changed
        rec['fix_changed_lines'] = len(candidate_lines)
        rec['fix_changed_set'] = sorted(candidate_lines)

        overlap = all_lines & candidate_lines
        rec['overlap_lines'] = len(overlap)
        rec['overlap_set'] = sorted(overlap)
        rec['overlap_rate'] = round(len(overlap) / len(candidate_lines), 3) if candidate_lines else 0.0
        rec['status'] = 'ok'
        results.append(rec)
        n_ok += 1
        print(f"[{idx_of.get(cid, '?'):>3}] ✓ {repo}#{pr} [{ext}] {cat}: fix={len(fix_changed)} "
              f"linter={len(all_lines)} overlap={len(overlap)}")

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"RQ3 Static Analysis Summary")
    print(f"{'='*70}")
    print(f"Cases: {len(results)}  ok={n_ok}  skip_no_linter={n_skip_lang}  "
          f"skip_file={n_skip_file}  skip_golangci={n_skip_go}  error={n_err}")

    ok = [r for r in results if r['status'] == 'ok']
    if ok:
        total_fix = sum(r['fix_changed_lines'] for r in ok)
        total_overlap = sum(r['overlap_lines'] for r in ok)
        print(f"\nAggregate for analyzable cases (n={len(ok)}):")
        print(f"  repair-changed lines: {total_fix}  overlap: {total_overlap}  "
              f"Match rate: {total_overlap}/{total_fix} = {total_overlap/total_fix*100:.1f}%"
              if total_fix else "  N/A")
        by_cat = defaultdict(list)
        for r in ok:
            by_cat[r['td_category']].append(r)
        print("\nMatch rate by TD category:")
        for c in sorted(by_cat):
            rs = by_cat[c]
            f = sum(x['fix_changed_lines'] for x in rs)
            o = sum(x['overlap_lines'] for x in rs)
            print(f"  {c:<6} n={len(rs):<3} {o}/{f} = {o/f*100:.0f}%" if f else f"  {c:<6} n={len(rs)}")

    print(f"\nOutput -> {out_file}")


if __name__ == '__main__':
    main()

"""Prepare PR-Agent replay inputs for the current 1,058-CTD / 531-PR frame."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/output/full_scale"
CURRENT = ROOT / "src/output/human_control/final_matching_current/ai_ctd_clustering_current_2717.json"
FLAT = OUT / "rq3_full_flat_manifest.json"
MEMBER_CONTEXTS = OUT / "rq3_full_contexts_budgeted.json"
OLD_PR_MANIFEST = OUT / "rq3_pr_level_manifest.json"
OLD_PR_CONTEXTS = OUT / "rq3_pr_level_contexts_budgeted.json"
OLD_PR_INDEX = OUT / "rq3_pr_level_head"
TARGET_MANIFEST = OUT / "rq3_current_1058_pr_replay_manifest.json"
TARGET_CONTEXTS = OUT / "rq3_current_1058_pr_replay_contexts.json"
TARGET_DIFFS = OUT / "rq3_current_1058_pr_diffs"
RUN_DIR = OUT / "rq3_current_1058_pr_replay"

sys.path.insert(0, str(ROOT / "src/rq3/scripts"))
from rq3_context_utils import (  # noqa: E402
    CONVENTION_PATHS, MAX_CONV_FILES, MAX_CONV_LINES, MAX_FILE_LINES,
    MAX_SIBLING_FILES, MAX_SIBLING_LINES, changed_new_ranges, merge_windows,
    render_window, truncate_lines, with_line_numbers,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   encoding="utf-8", errors="replace",
                                   stderr=subprocess.DEVNULL)


def language_for(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".go": "go", ".py": "python", ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript", ".vue": "vue", ".rs": "rust",
        ".java": "java", ".kt": "kotlin", ".php": "php", ".rb": "ruby",
        ".cs": "csharp", ".c": "c", ".h": "c", ".cpp": "cpp",
    }.get(ext, "")


def budget(text: str, maximum: int) -> str:
    if len(text) <= maximum:
        return text
    marker = f"\n\n[PR-LEVEL CONTEXT TRUNCATED BY PRE-SPECIFIED {maximum:,}-CHAR BUDGET; lower-priority suffix omitted.]\n"
    return text[:max(0, maximum - len(marker))] + marker


def git_context(pr: dict, idx: int) -> tuple[dict, dict]:
    repo_key = pr["pr_key"].split("#", 1)[0]
    repo_dir = ROOT / "repositories" / repo_key.split("/")[-1]
    merge = git(repo_dir, "log", "--all", "--format=%H", "--grep", f"pull request #{pr['pr_number']}", "-n", "1").strip()
    if not merge:
        merge = git(repo_dir, "log", "--all", "--format=%H", "--grep", pr.get("pr_title", ""), "-n", "1").strip()
    if not merge:
        raise RuntimeError(f"No merged commit found for {pr['pr_key']}")
    parent = git(repo_dir, "rev-parse", f"{merge}^1").strip()
    patch = git(repo_dir, "diff", "--no-ext-diff", "--find-renames", parent, merge)
    if not patch.strip():
        raise RuntimeError(f"Empty origin PR diff for {pr['pr_key']}")
    patch_path = TARGET_DIFFS / f"{idx:04d}_{re.sub(r'[^A-Za-z0-9]+', '_', pr['pr_key'])}.diff"
    patch_path.write_text(patch, encoding="utf-8")
    files = [x for x in git(repo_dir, "diff", "--name-only", "--find-renames", parent, merge).splitlines() if x]
    languages = list(dict.fromkeys(language_for(x) for x in files if language_for(x)))
    pr["diff_file"] = str(patch_path)
    pr["language"] = languages[0] if languages else ""
    pr["languages"] = languages
    pr["source_merge_sha"] = merge
    pr["source_base_sha"] = parent

    levels = {"L1": ""}
    all_file_contexts = []
    l2_parts = []
    l3_parts = []
    change_sizes = {}
    for filename in files:
        try:
            post = git(repo_dir, "show", f"{merge}:{filename}")
        except subprocess.CalledProcessError:
            continue
        file_patch = "\n".join(
            block for block in patch.split("diff --git ")
            if block.startswith(f"a/{filename} b/{filename}")
        )
        ranges = changed_new_ranges("diff --git " + file_patch if file_patch else "")
        lines = post.splitlines()
        if ranges:
            window = render_window(lines, merge_windows(ranges, 120, len(lines)))
            l2_parts.append(f"## Changed code context for `{filename}`\n\n{window}")
        trimmed, total = truncate_lines(post, MAX_FILE_LINES)
        body = f"## Full changed file: `{filename}` ({total} lines)\n\n{with_line_numbers(trimmed)}"
        l3_parts.append(body)
        all_file_contexts.append((filename, body))
        change_sizes[filename] = patch.count(f"+++ b/{filename}") + patch.count(f"--- a/{filename}")
    l2 = "\n\n---\n\n".join(l2_parts)
    target_files = set(pr.get("current_target_files", []))
    primary_parts = [body for filename, body in all_file_contexts if filename in target_files]
    if not primary_parts:
        primary_parts = l3_parts[:MAX_SIBLING_FILES]
    l3 = "\n\n---\n\n".join(primary_parts)
    levels["L2"] = l2
    levels["L3"] = l3
    siblings = [body for filename, body in sorted(all_file_contexts, key=lambda x: -change_sizes.get(x[0], 0))
                if filename not in {name for name, _ in all_file_contexts if name in target_files}][:MAX_SIBLING_FILES]
    levels["L4"] = l3 + ("\n\n## Other files changed in this PR\n" + "\n\n".join(siblings) if siblings else "")
    conventions = []
    for filename in CONVENTION_PATHS:
        if len(conventions) >= MAX_CONV_FILES:
            break
        try:
            text = git(repo_dir, "show", f"{merge}:{filename}")
        except subprocess.CalledProcessError:
            continue
        trimmed, total = truncate_lines(text, MAX_CONV_LINES)
        conventions.append(f"### `{filename}` ({total} lines)\n{with_line_numbers(trimmed)}")
    levels["L5"] = levels["L4"] + ("\n\n## Repository conventions / documentation\n" + "\n\n".join(conventions) if conventions else "")
    max_chars = {"L1": 0, "L2": 30_000, "L3": 50_000, "L4": 65_000, "L5": 80_000}
    final_levels = {level: budget(text, max_chars[level]) for level, text in levels.items()}
    context = {"idx": idx, "pr_key": pr["pr_key"], "levels": final_levels,
               "context_source": "local merged PR commit; before/after snapshots from repository clone"}
    return pr, context


def main() -> None:
    current_rows = read_json(CURRENT)["obligations"]
    if len(current_rows) != 1058:
        raise RuntimeError(f"Expected 1,058 current CTDs, found {len(current_rows)}")
    ct_ds_by_pr: OrderedDict[str, list[dict]] = OrderedDict()
    for current_ctd_index, row in enumerate(current_rows):
        row["current_ctd_index"] = current_ctd_index
        key = f"{row['repo']}#{row['pr']}"
        ct_ds_by_pr.setdefault(key, []).append(row)
    if len(ct_ds_by_pr) != 531:
        raise RuntimeError(f"Expected 531 current PRs, found {len(ct_ds_by_pr)}")

    current_case_ids = {case for row in current_rows for case in row["member_case_ids"]}
    flat = read_json(FLAT)
    flat_by_case = {row["case_id"]: row for row in flat}
    missing = current_case_ids - set(flat_by_case)
    old_manifest = read_json(OLD_PR_MANIFEST)
    old_contexts = {row["idx"]: row for row in read_json(OLD_PR_CONTEXTS)}
    old_by_key = {row["pr_key"]: row for row in old_manifest}
    target_manifest, target_contexts = [], []
    TARGET_DIFFS.mkdir(parents=True, exist_ok=True)
    added = 0

    for pr_key, ctd_rows in ct_ds_by_pr.items():
        if pr_key in old_by_key:
            old = old_by_key[pr_key]
            row = dict(old)
            row["current_ctd_indices"] = [int(x["current_ctd_index"]) for x in ctd_rows]
            row["current_member_case_ids"] = sorted({case for c in ctd_rows for case in c["member_case_ids"]})
            target_manifest.append(row)
            target_contexts.append(old_contexts[old["idx"]])
            continue
        added += 1
        first = ctd_rows[0]
        idx = len(old_manifest) + added
        member_ids = sorted({case for c in ctd_rows for case in c["member_case_ids"]})
        pr = {"idx": idx, "pr_key": pr_key, "repo": first["repo"], "pr_number": int(first["pr"]),
              "filepath": "(entire pull request)", "pr_title": first.get("pr_title", ""),
              "pr_body": "", "pr_merged_at": first.get("merged_at", ""),
              "n_members": len(member_ids), "current_ctd_indices": [int(x["current_ctd_index"]) for x in ctd_rows],
              "current_member_case_ids": member_ids,
              "current_target_files": sorted({filename for x in ctd_rows for filename in x.get("affected_files", [])}),
              "member_indices": []}
        pr, context = git_context(pr, idx)
        target_manifest.append(pr)
        target_contexts.append(context)

    # Reuse the five context-level reviews already completed for the 134 current
    # target PRs in the original stratified run.
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    current_keys = set(ct_ds_by_pr)
    copied = 0
    for row in target_manifest:
        old = old_by_key.get(row["pr_key"])
        if not old or row["pr_key"] not in current_keys:
            continue
        for level in ("L1", "L2", "L3", "L4", "L5"):
            source = OLD_PR_INDEX / f"{old['idx']:03d}_review_{level}.md"
            if source.is_file() and source.stat().st_size:
                shutil.copy2(source, RUN_DIR / f"{row['idx']:03d}_review_{level}.md")
                copied += 1

    TARGET_MANIFEST.write_text(json.dumps(target_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    TARGET_CONTEXTS.write_text(json.dumps(target_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "current_ctds": len(current_rows), "current_member_keys": len(current_case_ids),
        "current_prs": len(target_manifest), "matched_old_member_keys": len(current_case_ids & set(flat_by_case)),
        "unmatched_old_member_keys": len(missing), "prs_reconstructed_from_repository": added,
        "reused_existing_review_files": copied, "planned_review_calls": len(target_manifest) * 5 - copied,
        "manifest": str(TARGET_MANIFEST), "contexts": str(TARGET_CONTEXTS), "review_output_dir": str(RUN_DIR),
    }
    (OUT / "rq3_current_1058_pr_replay_prepare_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

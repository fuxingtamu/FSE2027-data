"""Semantically judge full current-frame PR-Agent reviews against 1,058 CTDs."""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/output/full_scale"
CURRENT = ROOT / "src/output/human_control/final_matching_current/ai_ctd_clustering_current_2717.json"
OBLIGATION_INDEX = OUT / "obligation_index.json"
MANIFEST = OUT / "rq3_current_1058_pr_replay_manifest.json"
REVIEWS = OUT / "rq3_current_1058_pr_replay"
RESULT = OUT / "rq3_current_1058_pr_replay_judgments.json"
PROGRESS = OUT / "rq3_current_1058_pr_replay_judge_progress.json"
MODEL = os.environ.get("V_LLM_MODEL", "gpt-5.6-luna")
BASE = (os.environ.get("V_LLM_BASE_URL") or "https://api.gpt.ge/v1/").rstrip("/") + "/"
SYSTEM = """You are an independent evaluator for a software-engineering study.
Judge whether a PR-Agent review identifies the same consequential defect,
omission, contract violation, or maintainability problem addressed by one of
the supplied later repairs from that PR. Shared file, line, topic, generic
advice, or a hypothetical concern is not enough. Return JSON only:
{"matched_ctd_indices": [integer, ...], "reasoning": "one concise sentence"}.
Use an empty list when no supplied CTD is matched. Only return indices listed
in the evidence."""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_keys() -> list[str]:
    keys = []
    for line in (ROOT / ".env").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if name.strip().startswith("V_OPENAI_API_KEY") and value:
            keys.append(value)
        elif name.strip() == "V_LLM_BASE_URL" and value:
            os.environ.setdefault("V_LLM_BASE_URL", value)
        elif name.strip() == "V_LLM_MODEL" and value:
            os.environ.setdefault("V_LLM_MODEL", value)
    return sorted(set(keys))


def git(repo: Path, *args: str) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   encoding="utf-8", errors="replace",
                                   stderr=subprocess.DEVNULL)


def main() -> None:
    keys = read_keys()
    if not keys:
        raise SystemExit("No V_OPENAI_API_KEY entries found")
    current = read_json(CURRENT)["obligations"]
    manifest = read_json(MANIFEST)
    if len(current) != 1058 or len(manifest) != 531:
        raise RuntimeError(f"Unexpected frame sizes: CTDs={len(current)}, PRs={len(manifest)}")
    index_data = read_json(OBLIGATION_INDEX)
    index_rows = [member for obligation in index_data["obligations"]
                  for member in obligation.get("members", [])]
    index_by_case = {row["case_id"]: row for row in index_rows}
    cases = {case for ctd in current for case in ctd["member_case_ids"]}
    missing = cases - set(index_by_case)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} current members from obligation_index")

    by_pr: dict[str, list[dict]] = {}
    for current_ctd_index, ctd in enumerate(current):
        ctd["current_ctd_index"] = current_ctd_index
        by_pr.setdefault(f"{ctd['repo']}#{ctd['pr']}", []).append(ctd)

    evidence_by_ctd: dict[int, str] = {}
    diff_cache: dict[tuple[str, str], str] = {}
    for ctd in current:
        pieces = []
        for case_id in ctd["member_case_ids"]:
            row = index_by_case[case_id]
            commit = row.get("commit_sha", "")
            repo = ROOT / "repositories" / str(row.get("repo_name", "")).split("/")[-1]
            filepath = row.get("filepath", "")
            cache_key = (commit, filepath)
            if cache_key not in diff_cache:
                try:
                    diff_cache[cache_key] = git(repo, "show", "--format=", "--no-ext-diff", commit,
                                                "--", filepath)
                except Exception:
                    diff_cache[cache_key] = ""
            diff = diff_cache[cache_key]
            if len(diff) > 2400:
                diff = diff[:2400] + "\n[repair diff truncated]"
            try:
                message = git(repo, "show", "-s", "--format=%s", commit).strip()
            except Exception:
                message = ""
            pieces.append(f"Member file: {row.get('filepath', '')}\nRepair commit: {message}\nRepair diff:\n{diff or '[repair diff unavailable]'}")
        evidence_by_ctd[int(ctd["current_ctd_index"])] = "\n\n".join(pieces)[:5000]

    tasks = []
    for pr in manifest:
        targets = by_pr.get(pr["pr_key"], [])
        if not targets:
            raise RuntimeError(f"No current CTDs for {pr['pr_key']}")
        target_text = []
        for ctd in targets:
            idx = int(ctd["current_ctd_index"])
            target_text.append(
                f"CTD index={idx}; category={ctd['category']}; obligation={ctd.get('description', '')}\n"
                f"Root cause: {ctd.get('root_cause', '')}\n"
                f"Affected files: {', '.join(ctd.get('affected_files', []))}\n"
                f"Later repair evidence:\n{evidence_by_ctd[idx]}"
            )
        for level in ("L1", "L2", "L3", "L4", "L5"):
            review_path = REVIEWS / f"{pr['idx']:03d}_review_{level}.md"
            if not review_path.is_file():
                raise RuntimeError(f"Missing review output: {review_path}")
            review = review_path.read_text(encoding="utf-8", errors="replace")
            prompt = (
                f"Origin PR: {pr['pr_key']}\nContext level: {level}\n\n"
                f"Current target CTDs and their later repair evidence:\n\n{'\n\n---\n\n'.join(target_text)}\n\n"
                f"PR-Agent review output:\n{review}\n\n"
                "Which current target CTDs, if any, did this review identify?"
            )
            tasks.append({"idx": pr["idx"], "pr_key": pr["pr_key"], "level": level,
                          "target_ctd_indices": [int(x["current_ctd_index"]) for x in targets],
                          "system_prompt": SYSTEM, "user_prompt": prompt})

    prior = {}
    for checkpoint in (PROGRESS, RESULT):
        if not checkpoint.exists():
            continue
        saved = read_json(checkpoint)
        if isinstance(saved, list):
            prior = {(x["idx"], x["level"]): x for x in saved
                     if x.get("matched_ctd_indices") is not None and x.get("error") is None}
            break
    pending = [task for task in tasks if (task["idx"], task["level"]) not in prior]
    results = list(prior.values())
    print(f"reviews={len(tasks)} complete={len(results)} pending={len(pending)}", flush=True)

    def run(task: dict, ordinal: int) -> dict:
        client = OpenAI(api_key=keys[ordinal % len(keys)], base_url=BASE,
                        timeout=180, max_retries=1)
        error = ""
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": task["user_prompt"]}],
                    temperature=0.0, max_tokens=1000,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
                parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
                allowed = set(task["target_ctd_indices"])
                matched = sorted({int(i) for i in parsed.get("matched_ctd_indices", []) if int(i) in allowed})
                return {k: task[k] for k in ("idx", "pr_key", "level", "target_ctd_indices")} | {
                    "matched_ctd_indices": matched, "detected": bool(matched),
                    "reasoning": parsed.get("reasoning", ""), "error": None}
            except Exception as exc:
                error = str(exc)[:300]
                if attempt < 3:
                    time.sleep(1 + attempt)
        return {k: task[k] for k in ("idx", "pr_key", "level", "target_ctd_indices")} | {
            "matched_ctd_indices": None, "detected": None, "reasoning": "", "error": error}

    with ThreadPoolExecutor(max_workers=min(10, len(keys))) as pool:
        futures = {pool.submit(run, task, i): task for i, task in enumerate(pending)}
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 20 == 0 or done == len(futures):
                results.sort(key=lambda x: (x["idx"], x["level"]))
                PROGRESS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"completed={done}/{len(futures)}", flush=True)

    results.sort(key=lambda x: (x["idx"], x["level"]))
    RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {}
    for level in ("L1", "L2", "L3", "L4", "L5"):
        rows = [x for x in results if x["level"] == level]
        valid = [x for x in rows if x.get("detected") is not None]
        hits = [x for x in valid if x["detected"]]
        target_ctds = {idx for row in rows for idx in row["target_ctd_indices"]}
        matched_ctds = {idx for row in valid for idx in (row.get("matched_ctd_indices") or [])}
        summary[level] = {
            "reviews": len(rows), "valid_reviews": len(valid), "hit_reviews": len(hits),
            "pr_hit_rate_percent": round(100 * len(hits) / max(1, len(valid)), 2),
            "target_ctds": len(target_ctds), "matched_ctds": len(matched_ctds),
            "ctd_hit_rate_percent": round(100 * len(matched_ctds) / max(1, len(target_ctds)), 2),
        }
    progress = {"stage": "completed", "reviews": len(tasks), "completed": len(results),
                "unresolved": sum(x.get("detected") is None for x in results), "summary": summary}
    PROGRESS.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

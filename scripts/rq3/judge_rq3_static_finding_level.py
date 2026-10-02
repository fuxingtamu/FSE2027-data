"""Classify static-analysis diagnostic instances against later CTD repairs."""
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
STATIC = OUT / "rq3_static_merge_snapshot.json"
PACKETS = OUT / "rq2_full_obligation_packets.json"
RESULT = OUT / "rq3_static_merge_snapshot_finding_judgments.json"
PROGRESS = OUT / "rq3_static_merge_snapshot_finding_judgments.progress.json"
MODEL_DEFAULT = "gpt-5.6-luna"
BASE_DEFAULT = "https://api.gpt.ge/v1/"
CHUNK_SIZE = 25

SYSTEM = """You are a software-engineering researcher evaluating static-analysis
diagnostics against a later validated repair. Classify each supplied diagnostic
independently. A match means the diagnostic identifies the same underlying
defect, omission, contract violation, or maintainability deficiency addressed
by the repair. A diagnostic on a changed line is only a candidate: unrelated
formatting, naming, import, style, or another issue is not a match unless it
explains the repaired deficiency. Return JSON only:
{"matched_finding_ids": [integer, ...], "reasoning": "brief explanation"}.
Include every supplied finding ID that matches; return an empty list if none do."""


def load_keys() -> list[str]:
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
        elif name.strip() in {"V_LLM_BASE_URL", "V_LLM_MODEL"} and value:
            os.environ.setdefault(name.strip(), value)
    return sorted(set(keys))


def fix_text(member: dict) -> str:
    value = member.get("fix_diff") or {}
    if isinstance(value, dict):
        value = value.get("diff", "")
    return str(value or "")


def candidate_findings(row: dict) -> list[dict]:
    overlap = {int(x) for x in row.get("overlap_set", [])}
    findings = []
    for tool, line_map in (row.get("linter_details") or {}).items():
        if tool == "__error__":
            continue
        for line_text, diagnostics in line_map.items():
            line = int(line_text)
            if line not in overlap:
                continue
            for occurrence, diagnostic in enumerate(diagnostics, 1):
                findings.append({
                    "finding_id": len(findings) + 1,
                    "tool": tool,
                    "line": line,
                    "occurrence": occurrence,
                    "diagnostic": str(diagnostic),
                    "hit": None,
                    "reasoning": "",
                })
    return findings


def main() -> None:
    keys = load_keys()
    if not keys:
        raise SystemExit("No V_OPENAI_API_KEY entries found in .env")
    model = os.environ.get("V_LLM_MODEL", MODEL_DEFAULT)
    base = os.environ.get("V_LLM_BASE_URL", BASE_DEFAULT).rstrip("/") + "/"

    current = json.loads(CURRENT.read_text(encoding="utf-8"))["obligations"]
    ctd_by_case = {}
    for index, ctd in enumerate(current):
        item = {"index": index, **ctd}
        for case_id in ctd["member_case_ids"]:
            ctd_by_case[case_id] = item

    packets = json.loads(PACKETS.read_text(encoding="utf-8"))
    member_by_case = {
        member["case_id"]: member
        for obligation in packets["obligations"]
        for member in obligation.get("members", [])
    }
    static_rows = json.loads(STATIC.read_text(encoding="utf-8"))
    target_cases = set(ctd_by_case)
    rows = [r for r in static_rows
            if r.get("case_id") in target_cases and r.get("status") == "ok"]
    all_findings = []
    pending_tasks = []
    no_overlap_count = 0
    for row in rows:
        ctd = ctd_by_case[row["case_id"]]
        findings = []
        for tool, line_map in (row.get("linter_details") or {}).items():
            if tool == "__error__":
                continue
            for line_text, diagnostics in line_map.items():
                line = int(line_text)
                for occurrence, diagnostic in enumerate(diagnostics, 1):
                    findings.append({
                        "finding_id": f"{row['case_id']}|{tool}|{line}|{occurrence}",
                        "case_id": row["case_id"], "ctd_index": ctd["index"],
                        "pr_key": f"{ctd['repo']}#{ctd['pr']}",
                        "filepath": row.get("filepath", ""),
                        "tool": tool, "line": line, "occurrence": occurrence,
                        "diagnostic": str(diagnostic),
                        "on_repair_changed_line": line in set(row.get("overlap_set", [])),
                        "hit": None, "reasoning": "",
                    })
        all_findings.extend(findings)
        candidate = [f for f in findings if f["on_repair_changed_line"]]
        for f in findings:
            if not f["on_repair_changed_line"]:
                f["hit"] = False
                f["reasoning"] = "Diagnostic does not overlap a repair-changed line; excluded by the study's candidate rule."
                no_overlap_count += 1
        # Judge each repair-line candidate independently, batching findings
        # that share the same CTD repair.
        ctd_text = (
            f"CTD {ctd['index']} ({ctd.get('category', '')}, {ctd.get('subtype', '')})\n"
            f"Description: {ctd.get('description', '')}\n"
            f"Root cause: {ctd.get('root_cause', '')}\n"
        )
        member = member_by_case.get(row["case_id"], {})
        diff = fix_text(member)
        if len(diff) > 9000:
            diff = diff[:9000] + "\n[repair diff truncated]"
        candidate.sort(key=lambda f: f["finding_id"])
        for start in range(0, len(candidate), CHUNK_SIZE):
            chunk = candidate[start:start + CHUNK_SIZE]
            finding_lines = "\n".join(
                f"ID {i}: line {f['line']} [{f['tool']}] {f['diagnostic']}"
                for i, f in enumerate(chunk)
            )
            prompt = (
                f"PR: {ctd['repo']}#{ctd['pr']}\nFile: {row.get('filepath', '')}\n"
                "The diagnostics were produced from this file in the PR snapshot immediately before merge.\n\n"
                f"Later-repaired target CTD:\n{ctd_text}\n"
                f"Repair commit: {member.get('commit_message', '')}\n"
                f"Repair diff:\n```diff\n{diff}\n```\n\n"
                "Classify each finding independently. Only findings on the repair-changed "
                "lines are listed below. Return the IDs of all and only findings that "
                "identify the same underlying problem as this repair.\n\n"
                f"Candidate findings:\n{finding_lines}"
            )
            pending_tasks.append({
                "case_id": row["case_id"], "ctd_index": ctd["index"],
                "pr_key": f"{ctd['repo']}#{ctd['pr']}",
                "finding_ids": [f["finding_id"] for f in chunk],
                "prompt": prompt,
            })

    by_id = {f["finding_id"]: f for f in all_findings}
    existing_results = {}
    if PROGRESS.exists():
        existing_results = {
            (tuple(x["finding_ids"])): x
            for x in json.loads(PROGRESS.read_text(encoding="utf-8"))
            if x.get("hit_ids") is not None
        }
    pending_tasks = [t for t in pending_tasks if tuple(t["finding_ids"]) not in existing_results]
    print(
        f"target_ctds={len(current)} successful_member_runs={len(rows)} "
        f"diagnostics={len(all_findings)} candidates={sum(f['on_repair_changed_line'] for f in all_findings)} "
        f"auto_no_overlap={no_overlap_count} "
        f"item_level_batches={len(pending_tasks)} prior_batches={len(existing_results)}",
        flush=True,
    )

    def run(task: dict, ordinal: int) -> dict:
        client = OpenAI(api_key=keys[ordinal % len(keys)], base_url=base,
                        timeout=180, max_retries=1)
        error = ""
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": task["prompt"]}],
                    temperature=0.0, max_tokens=1200,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
                if not content:
                    raise ValueError(f"empty response, finish_reason={response.choices[0].finish_reason}")
                parsed = json.loads(content)
                allowed = {str(i) for i in range(len(task["finding_ids"]))}
                # The prompt uses local zero-based IDs; map them back to stable IDs.
                hit_local = {str(i) for i in parsed.get("matched_finding_ids", [])}
                hit_ids = [task["finding_ids"][int(i)] for i in hit_local
                           if i in allowed]
                return {"case_id": task["case_id"], "ctd_index": task["ctd_index"],
                        "pr_key": task["pr_key"], "finding_ids": task["finding_ids"],
                        "hit_ids": hit_ids, "reasoning": parsed.get("reasoning", ""),
                        "error": None}
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:300]
                if attempt < 4:
                    time.sleep(1 + attempt)
        return {"case_id": task["case_id"], "ctd_index": task["ctd_index"],
                "pr_key": task["pr_key"], "finding_ids": task["finding_ids"],
                "hit_ids": None, "reasoning": "", "error": error}

    results = list(existing_results.values())
    with ThreadPoolExecutor(max_workers=min(10, len(keys))) as pool:
        futures = {pool.submit(run, task, i): task for i, task in enumerate(pending_tasks)}
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 10 == 0 or done == len(futures):
                results.sort(key=lambda x: x["finding_ids"])
                PROGRESS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"completed_batches={done}/{len(futures)}", flush=True)

    task_results = {finding_id: result for result in results
                    for finding_id in result["finding_ids"]}
    unresolved = 0
    for f in all_findings:
        if f["hit"] is not None:
            continue
        result = task_results.get(f["finding_id"])
        if result is None or result.get("hit_ids") is None:
            unresolved += 1
            continue
        f["hit"] = f["finding_id"] in result["hit_ids"]
        f["reasoning"] = result.get("reasoning", "")

    summary = {
        "scope": "all individual diagnostics from successful static-analysis runs on current 1,058 CTDs",
        "model": model,
        "target_ctds": len(current),
        "successful_member_runs": len(rows),
        "diagnostic_count": len(all_findings),
        "repair_line_candidate_count": sum(f["on_repair_changed_line"] for f in all_findings),
        "nonoverlap_diagnostics_counted_as_no_match": no_overlap_count,
        "unresolved_diagnostics": unresolved,
        "matched_diagnostics": sum(f.get("hit") is True for f in all_findings),
        "not_matched_diagnostics": sum(f.get("hit") is False for f in all_findings),
        "match_rate_percent": round(100 * sum(f.get("hit") is True for f in all_findings) / len(all_findings), 4),
    }
    RESULT.write_text(json.dumps({"summary": summary, "findings": all_findings},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"saved={RESULT}", flush=True)


if __name__ == "__main__":
    main()



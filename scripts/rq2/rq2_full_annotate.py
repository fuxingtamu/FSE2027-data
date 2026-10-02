"""Obligation-level RQ2 annotation for the full-scale evidence packets.

The script is resumable and supports ``--limit`` for a task-level smoke test.
It deliberately emits one annotation per independent obligation, not one per
file member or repair commit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKETS = ROOT / "src/output/full_scale/rq2_full_obligation_packets.json"
EVOLUTION = ROOT / "src/output/full_scale/rq2_full_evolution_evidence.json"
OUT = ROOT / "src/output/full_scale"
MODEL = "gpt-5.6-luna"
ENV = ROOT / ".env"


def load_env():
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
BASE_URL = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
KEYS = [os.environ[k] for k in sorted(os.environ) if k.startswith("OPENAI_API_KEY") and os.environ[k]]

SYSTEM = """You are a software engineering empirical researcher. Annotate one consequential technical-debt obligation using only the public evidence supplied. The unit is the obligation, which may span several files in an AI-attributed PR. Do not infer private developer intent. Return JSON only.

Exposure labels: EX1=user-reported failure or report after merge that substantively corresponds to the repair; EX2=explicit automated failure signal; EX3=post-merge code evolution that makes the implementation insufficient; EX4=internal maintenance or cleanup without an external trigger; EXU=public evidence is insufficient.
Root-cause labels: RC1=AI-local omission of an obligation within the PR's intended behavior; RC2=AI-local implemented logic is functionally inadequate; RC3=quality or repository-convention gap without demonstrated functional failure; RC4=post-merge context change makes the implementation insufficient; RCU=insufficient evidence.
Repair labels: RP1=guard/check/error handling; RP2=logic/contract/type correction; RP3=refactor/delete; RP4=style/docs/convention; RP5=test/tooling; RP6=dependency/configuration.

Use evidence_state=\"evidence present\", \"evidence absent\", \"artifact unavailable\", or \"cannot determine\". Use confidence=high, medium, or low. Keep rationale concise and evidence-based."""


def call(item: dict, retries: int = 4) -> dict:
    def text(value, limit):
        if value is None:
            return ""
        if isinstance(value, str):
            return value[:limit]
        return json.dumps(value, ensure_ascii=False)[:limit]

    members = item.get("members", [])
    member_text = []
    for m in members[:12]:
        member_text.append(
            f"FILE {m.get('filepath')} commit={m.get('commit_sha','')[:10]}\n"
            f"MERGED={m.get('pr_merged_at')} COMMIT_DATE={m.get('commit_date')} "
            f"FIX_LATENCY_DAYS={m.get('fix_latency_days')}\n"
            f"COMMIT_MESSAGE:\n{text(m.get('commit_message'), 1000)}\n"
            f"LINKED_ISSUES:\n{text(m.get('linked_issues'), 2400)}\n"
            f"ISSUE_REFS_FOUND:\n{text(m.get('issue_refs_found'), 1600)}\n"
            f"FIX_COMMIT_INFO:\n{text(m.get('fix_commit_info'), 1800)}\n"
            f"FIX_CATEGORY={m.get('fix_category')} FIX_IS_RELEASE={m.get('fix_is_release')}\n"
            f"PATCH:\n{text(m.get('pr_file_patch'), 2200)}\n"
            f"FIX DIFF:\n{text(m.get('fix_diff'), 1800)}"
        )
    prompt = {
        "obligation": {k: item.get(k) for k in ("obligation_id", "repo", "pr", "category", "subtype", "nature", "description", "root_cause", "affected_files", "repayment_commits")},
        "pr_context": [{k: item["members"][0].get(k) for k in ("pr_title", "pr_body", "pr_merged_at", "agent")} ] if members else [],
        "evidence": "\n\n".join(member_text),
        "evolution_evidence": item.get("evolution_evidence", {}),
        "task": "Return {exposure, root_cause, repair_response, evidence_state, confidence, exposure_rationale, root_cause_rationale, repair_rationale}.",
    }
    payload = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}], "temperature": 0.0, "max_tokens": 900}
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {random.choice(KEYS)}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                body = json.loads(response.read())
            content = body["choices"][0]["message"]["content"]
            match = re.search(r"\{[\s\S]*\}", content)
            result = json.loads(match.group()) if match else {"error": "non_json_response"}
            result.update({"obligation_id": item["obligation_id"], "repo": item.get("repo"), "pr": item.get("pr")})
            return result
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
    return {"obligation_id": item["obligation_id"], "repo": item.get("repo"), "pr": item.get("pr"), "error": str(last)[:240]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["exposure", "rootcause", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="annotate only the first N obligations")
    ap.add_argument("--workers", type=int, default=max(1, len(KEYS)))
    ap.add_argument("--force", action="store_true", help="rerun selected obligations even if output exists")
    args = ap.parse_args()
    if not BASE_URL or not KEYS:
        raise SystemExit("missing LLM_BASE_URL or OPENAI_API_KEY*")
    data = json.loads(PACKETS.read_text(encoding="utf-8"))
    evolution = json.loads(EVOLUTION.read_text(encoding="utf-8"))
    evolution_by_id = {x["obligation_id"]: x for x in evolution["obligations"]}
    items = []
    for item in data["obligations"][:args.limit or None]:
        item = dict(item)
        item["evolution_evidence"] = evolution_by_id.get(item["obligation_id"], {})
        items.append(item)
    passes = [args.only] if args.only != "both" else ["both"]
    for pass_name in passes:
        path = OUT / f"rq2_full_annotations_{pass_name}.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        done = set() if args.force else {r.get("obligation_id") for r in existing if not r.get("error")}
        todo = [x for x in items if x["obligation_id"] not in done]
        print(f"pass={pass_name} total={len(items)} done={len(done)} todo={len(todo)} workers={args.workers}")
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(call, item): item for item in todo}
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    existing = [x for x in existing if x.get("obligation_id") != result.get("obligation_id")]
                    existing.append(result)
                    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                print(result.get("obligation_id"), result.get("exposure", result.get("error", "ok")), flush=True)


if __name__ == "__main__":
    main()

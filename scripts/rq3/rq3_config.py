"""Shared paths, environment loading, and PR-Agent configuration for RQ3."""
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", PACKAGE_ROOT.parent)).resolve()
DATA_DIR = Path(os.environ.get("RQ3_DATA_DIR", PACKAGE_ROOT / "data")).resolve()
WORK_DIR = Path(os.environ.get("RQ3_WORK_DIR", PACKAGE_ROOT / "work" / "rq3")).resolve()
ENV_FILE = Path(os.environ.get("ENV_FILE", PROJECT_ROOT / ".env")).resolve()
REPOS_DIR = Path(os.environ.get("REPOSITORIES_DIR", PROJECT_ROOT / "repositories")).resolve()
PR_AGENT_DIR = Path(os.environ.get("PR_AGENT_DIR", PROJECT_ROOT / "tools" / "pr-agent")).resolve()

USE_FULL_SCALE = os.environ.get("RQ3_FULL_SCALE", "1") == "1"
BASE_URL = (os.environ.get("V_LLM_BASE_URL") or "https://api.gpt.ge/v1/").rstrip("/")
MODEL = os.environ.get("V_LLM_MODEL") or "gpt-5.6-luna"

MERGED_FILE = Path(os.environ.get("RQ3_MERGED_FILE", DATA_DIR / "rq3_static_input_cases.json")).resolve()
MANIFEST_FILE = Path(os.environ.get("RQ3_MANIFEST_FILE", DATA_DIR / "rq3_static_manifest.json")).resolve()
DIFF_DIR = Path(os.environ.get("RQ3_DIFF_DIR", DATA_DIR / "rq3_diffs")).resolve()
CONTEXT_FILE = Path(os.environ.get("RQ3_CONTEXT_FILE", DATA_DIR / "rq3_contexts.json")).resolve()
REPLAY_MANIFEST_FILE = Path(os.environ.get("RQ3_REPLAY_MANIFEST_FILE", DATA_DIR / "rq3_pr_review_manifest.json")).resolve()
REPLAY_CONTEXT_FILE = Path(os.environ.get("RQ3_REPLAY_CONTEXT_FILE", DATA_DIR / "rq3_pr_review_contexts.json")).resolve()
COMMITS_CACHE = Path(os.environ.get("RQ3_COMMITS_CACHE", WORK_DIR / "pr_commits.json")).resolve()
RUN_OUT_DIR = Path(os.environ.get("RQ3_RUN_OUT_DIR", WORK_DIR / "pr_agent_reviews")).resolve()
STATUS_FILE = Path(os.environ.get("RQ3_STATUS_FILE", WORK_DIR / "pr_agent_status.json")).resolve()
DETECT_FILE = Path(os.environ.get("RQ3_DETECT_FILE", WORK_DIR / "pr_agent_judgments.json")).resolve()
STATIC_FILE = Path(os.environ.get("RQ3_STATIC_FILE", WORK_DIR / "static_analysis.json")).resolve()
STATIC_JUDGE_FILE = Path(os.environ.get("RQ3_STATIC_JUDGE_FILE", WORK_DIR / "static_analysis_judgments.json")).resolve()

REVIEW_LEVELS = ["L1", "L2", "L3", "L4", "L5"]
TASKS = [("review", level) for level in REVIEW_LEVELS]
OPTIONAL_TASKS = [("improve", "L3"), ("ask", "L5"), ("ask", "L1")]
ASK_QUESTION = (
    "Identify the technical debt introduced by this PR, including but not limited to "
    "missing error handling or boundary checks, missing null/undefined guards, "
    "overly broad or unsafe types, interface or abstraction leaks, duplicated code, "
    "and structural coupling. For each issue, cite the specific code location and "
    "describe a concrete future rework scenario it could cause."
)


def load_keys() -> list[str]:
    """Load configured API keys from the environment file."""
    keys = []
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            body = line[len("export "):].strip() if line.startswith("export ") else line
            prefix = "V_OPENAI_API_KEY" if USE_FULL_SCALE else "OPENAI_API_KEY"
            if body.startswith(prefix) and "=" in body:
                key = body.split("=", 1)[1].strip().strip("\"").strip("'")
                if key:
                    keys.append(key)
    prefix = "V_OPENAI_API_KEY" if USE_FULL_SCALE else "OPENAI_API_KEY"
    keys.extend(value for name, value in os.environ.items()
                if name.startswith(prefix) and value)
    return sorted(set(keys))


def setup_pr_agent(key: str) -> None:
    """Configure the PR-Agent settings for the selected compatible model."""
    from pr_agent.config_loader import get_settings
    settings = get_settings()
    settings.set("openai.key", key)
    settings.set("openai.api_base", BASE_URL)
    settings.set("config.model", MODEL)
    settings.set("config.model_weak", MODEL)
    settings.set("config.model_turbo", MODEL)
    settings.set("config.temperature", 0.0)
    settings.set("config.custom_model_max_tokens", 32000)
    settings.set("config.seed", -1)
    settings.set("config.publish_output", True)


if __name__ == "__main__":
    keys = load_keys()
    print(f"Configured API keys: {len(keys)}")
    print(f"Task matrix: {TASKS} ({len(TASKS)} tasks per case)")

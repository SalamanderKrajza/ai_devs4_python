import json
from datetime import datetime, UTC
from pathlib import Path


MODEL_PRICING_USD_PER_MILLION = {
    "gemini-3.1-flash-lite-preview": {
        "input": 0.25,
        "cached_input": 0.025,
        "output": 1.50,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "cached_input": 0.125,
        "output": 10.00,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "cached_input": 0.03,
        "output": 2.50,
    },
    "gemini-3-flash-preview": {
        "input": 0.50,
        "cached_input": 0.05,
        "output": 3.00,
    },
}


def create_run_logs_dir(base_dir: Path, run_prefix: str) -> tuple[Path, str]:
    """Create a per-run logs directory and return the directory with run id."""
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_prefix}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = logs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def extract_gemini_usage_metrics(response: object) -> dict:
    """Extract token usage data from Gemini response metadata."""
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "input_tokens": int(getattr(usage_metadata, "prompt_token_count", 0) or 0),
        "cached_input_tokens": int(
            getattr(usage_metadata, "cached_content_token_count", 0) or 0
        ),
        "output_tokens": int(getattr(usage_metadata, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(usage_metadata, "total_token_count", 0) or 0),
    }


def calculate_usage_cost_usd(model_name: str, usage_metrics: dict) -> float:
    """Estimate usage cost in USD using the configured per-model pricing table."""
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model_name)
    if pricing is None:
        return 0.0

    cached_input_tokens = usage_metrics["cached_input_tokens"]
    input_tokens = max(0, usage_metrics["input_tokens"] - cached_input_tokens)
    output_tokens = usage_metrics["output_tokens"]

    return (
        (input_tokens / 1_000_000) * pricing["input"]
        + (cached_input_tokens / 1_000_000) * pricing["cached_input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )


def create_usage_summary(model_name: str) -> dict:
    """Create an empty usage summary structure for one task run."""
    return {
        "model": model_name,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "actions_logged": 0,
    }


def append_usage_log(
    log_path: Path,
    usage_summary: dict,
    action: str,
    payload: dict,
    *,
    model: str | None = None,
    usage_metrics: dict | None = None,
    cost_usd: float = 0.0,
) -> None:
    """Append one structured log entry and update the shared usage summary."""
    usage_metrics = usage_metrics or {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": action,
        "model": model,
        "usage": usage_metrics,
        "cost_usd": cost_usd,
        "payload": payload,
    }
    with open(log_path, "a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    usage_summary["actions_logged"] += 1
    usage_summary["input_tokens"] += usage_metrics["input_tokens"]
    usage_summary["cached_input_tokens"] += usage_metrics["cached_input_tokens"]
    usage_summary["output_tokens"] += usage_metrics["output_tokens"]
    usage_summary["cost_usd"] += cost_usd

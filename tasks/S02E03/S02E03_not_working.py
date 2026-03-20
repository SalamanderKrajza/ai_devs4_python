import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from tasks.commons.llm_usage import (
    append_usage_log,
    calculate_usage_cost_usd,
    create_run_logs_dir,
    create_usage_summary,
)


# ==============================================================================
# S02E03 - Failure - Compressing a large system log into a diagnostic timeline
# Phase 0: Configuration and environment
# ==============================================================================

load_dotenv()

AI_DEVS_API_KEY: str = os.environ["AI_DEVS_API_KEY"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BASE_URL = "https://hub.ag3nts.org"
TASK = "failure"
LOG_URL = f"{BASE_URL}/data/{AI_DEVS_API_KEY}/failure.log"
VERIFY_URL = f"{BASE_URL}/verify"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_LOG_PATH = DATA_DIR / "failure.log"
SEVERITY_PROFILE_PATH = DATA_DIR / "severity_profile.json"
FINAL_LOGS_PATH = DATA_DIR / "final_logs.txt"
FEEDBACK_PATH = DATA_DIR / "last_feedback.json"
FOCUS_TERMS_PATH = DATA_DIR / "focus_terms.json"
ITERATION_SUMMARY_PATH = DATA_DIR / "iteration_summary.json"
RESULT_PATH = DATA_DIR / "result.txt"

RUN_LOGS_DIR, RUN_ID = create_run_logs_dir(DATA_DIR, "s02e03")
AGENT_LOG_PATH = RUN_LOGS_DIR / "agent_log.jsonl"

TOKEN_LIMIT = 1470  # conservative margin — server tokenizer may count ~2% more
MAX_VERIFY_ITERATIONS = 6
CHUNK_SIZE = 80
FORMAT_SAMPLE_LINES = 60

enc = tiktoken.get_encoding("o200k_base")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
run_cost_summary = create_usage_summary(OPENAI_MODEL)


# ------------------------------------------------------------------------------
# Phase 1: Utility helpers
# ------------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def extract_flag(payload: dict | str) -> str | None:
    text = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    match = re.search(r"\{FLG:[^}]+\}", text)
    return match.group(0) if match else None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for line in lines:
        if line not in seen:
            result.append(line)
            seen.add(line)
    return result


def build_submission_text(lines: list[str]) -> str:
    return "\n".join(lines)


def deduplicate_logs_by_message(lines: list[str]) -> list[str]:
    """Keep first occurrence of each unique message, ignoring the timestamp prefix."""
    timestamp_pattern = re.compile(r'^\[?\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}(:\d{2})?\]?\s*')
    seen_messages: set[str] = set()
    result = []
    for line in lines:
        message_key = timestamp_pattern.sub("", line).strip()
        if message_key not in seen_messages:
            seen_messages.add(message_key)
            result.append(line)
    return result


def extract_openai_usage_metrics(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "total_tokens": int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0),
    }


def prioritize_lines(lines: list[str], focus_terms: set[str]) -> list[tuple[int, str]]:
    """Score lines by focus term hits for budget trimming."""
    result = []
    for line in lines:
        lower = line.lower()
        score = sum(1 for term in focus_terms if term.lower() in lower)
        result.append((score, line))
    result.sort(key=lambda x: -x[0])
    return result


def trim_lines_to_token_limit(
    lines: list[str],
    token_limit: int,
    focus_terms: set[str],
) -> list[str]:
    unique = deduplicate_lines(lines)
    positions = {line: i for i, line in enumerate(unique)}
    prioritized = prioritize_lines(unique, focus_terms)
    selected: list[str] = []
    for _, line in prioritized:
        candidate = build_submission_text(
            sorted(selected + [line], key=lambda l: positions[l])
        )
        if count_tokens(candidate) <= token_limit:
            selected.append(line)
    selected.sort(key=lambda l: positions[l])
    return selected


def extract_timestamp_key(line: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2})", line)
    return match.group(1) if match else ""


def sort_lines_chronologically(lines: list[str]) -> list[str]:
    return sorted(lines, key=extract_timestamp_key)


def extract_feedback_focus_terms(feedback_payload: dict | str) -> list[str]:
    """Extract component-like uppercase tokens from hub feedback."""
    text = (
        json.dumps(feedback_payload, ensure_ascii=False)
        if isinstance(feedback_payload, dict)
        else str(feedback_payload)
    )
    return sorted(set(re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", text)))


# ------------------------------------------------------------------------------
# Phase 2: LLM-driven log format analysis
# ------------------------------------------------------------------------------

def analyze_log_format(sample_lines: list[str]) -> dict[str, Any]:
    """Ask LLM to identify severity patterns from a sample of log lines.

    Returns severity_keywords the agent will use for grep-style filtering —
    no assumptions about log format are hardcoded here.
    """
    sample_text = "\n".join(sample_lines)
    messages = [
        {
            "role": "system",
            "content": (
                "You analyze system log files to identify important events.\n"
                "Given a sample, return JSON with:\n"
                "- severity_keywords: list of strings (case-insensitive substrings) that appear "
                "in lines indicating high-severity or operationally important events "
                "(errors, warnings, failures, faults, trips, shutdowns, etc.)\n"
                "- format_note: one sentence describing the log line format\n"
                "Be inclusive — prefer recall over precision. Include all severity indicators you see.\n"
                "Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": f"Log sample ({len(sample_lines)} lines):\n\n{sample_text}",
        },
    ]
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    response_text = response.choices[0].message.content.strip()
    usage_metrics = extract_openai_usage_metrics(response)
    cost_usd = calculate_usage_cost_usd(OPENAI_MODEL, usage_metrics)
    append_usage_log(
        AGENT_LOG_PATH, run_cost_summary, "analyze_log_format",
        {"sample_line_count": len(sample_lines), "response": response_text},
        model=OPENAI_MODEL, usage_metrics=usage_metrics, cost_usd=cost_usd,
    )

    profile = json.loads(response_text)

    severity_keywords = [
        kw.strip() for kw in profile.get("severity_keywords", [])
        if isinstance(kw, str) and kw.strip()
    ]
    # Fallback: universal substrings present in virtually all log formats
    if not severity_keywords:
        severity_keywords = ["warn", "error", "err", "crit", "fatal", "fault", "fail", "alert", "trip", "shutdown"]

    return {
        "severity_keywords": severity_keywords,
        "format_note": profile.get("format_note", ""),
    }


def filter_lines_by_keywords(lines: list[str], keywords: list[str]) -> list[str]:
    """Keep lines that contain any of the keywords (case-insensitive)."""
    lower_keywords = [kw.lower() for kw in keywords]
    return [line for line in tqdm(lines, desc="Filtering lines") if any(kw in line.lower() for kw in lower_keywords)]


# ------------------------------------------------------------------------------
# Phase 3: LLM log reducer
# ------------------------------------------------------------------------------

def call_openai_log_reducer(
    candidate_lines: list[str],
    *,
    focus_terms: set[str],
    feedback_payload: dict | str | None,
    format_note: str,
    target_token_limit: int,
    action_name: str,
) -> list[str]:
    """Ask an LLM to compress log lines while preserving diagnostic signal."""
    if not candidate_lines:
        return []

    feedback_text = (
        json.dumps(feedback_payload, ensure_ascii=False)
        if isinstance(feedback_payload, dict)
        else str(feedback_payload or "")
    )
    reduced_lines: list[str] = []

    for chunk_index in tqdm(
        range(0, len(candidate_lines), CHUNK_SIZE),
        desc=f"{action_name}_chunks",
    ):
        chunk_lines = candidate_lines[chunk_index: chunk_index + CHUNK_SIZE]
        chunk_text = "\n".join(chunk_lines)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are compressing system failure logs for technicians.\n"
                    f"Log format: {format_note}\n"
                    "Return only the events required to diagnose the outage.\n"
                    "Output: one event per line, keeping timestamp, severity, component id and short message.\n"
                    "Do not add explanations, bullets or markdown.\n"
                    "Prefer concise paraphrases over long original text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Focus terms from technician feedback (Make sure that you will keep them in final file): {sorted(focus_terms)}\n"
                    f"Technician feedback: {feedback_text}\n"
                    f"Target budget: {target_token_limit} tokens total.\n"
                    "Reduce this chunk to highest-signal events only:\n\n"
                    f"{chunk_text}"
                ),
            },
        ]
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=messages,
        )
        response_text = response.choices[0].message.content.strip()
        usage_metrics = extract_openai_usage_metrics(response)
        cost_usd = calculate_usage_cost_usd(OPENAI_MODEL, usage_metrics)
        append_usage_log(
            AGENT_LOG_PATH, run_cost_summary, action_name,
            {
                "chunk_index": chunk_index // CHUNK_SIZE,
                "candidate_line_count": len(chunk_lines),
                "response_preview": response_text[:500],
            },
            model=OPENAI_MODEL, usage_metrics=usage_metrics, cost_usd=cost_usd,
        )
        chunk_output_lines = [
            normalize_whitespace(line)
            for line in response_text.splitlines()
            if normalize_whitespace(line)
        ]
        reduced_lines.extend(chunk_output_lines)

    return deduplicate_lines(reduced_lines)


# ------------------------------------------------------------------------------
# Phase 4: Verify
# ------------------------------------------------------------------------------

def verify_logs(logs_text: str) -> dict[str, Any]:
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"logs": logs_text},
    }
    response = requests.post(VERIFY_URL, json=payload, timeout=60)
    response.raise_for_status() if response.status_code >= 500 else None
    return response.json()


# ==============================================================================
# Main pipeline
# ==============================================================================
"""
NOTE:
This endpoint works weird from my perspective

I tested few flows with llm at first it started passing and getting new requests
for keywords, until I got them all
Then it complains about order
After sorting (without changing logs) it once again, complain for missing flags

I debuged this flow with maunaly addeed logs to check how system works with something like this:
x = '[2026-03-19 06:37:33] [ERRO] STMTURB12 feedback loop exceeded correction budget. Thermal conversion rate is reduced \n' * 20
x+= '[2026-03-19 06:36:40] [ERRO] WTANK07 indicates unstable refill trend. Available coolant inventory is no longer guaranteed.\n'
x+= '[2026-03-19 08:36:50] [ERRO] WTANK07 level estimate dropped near minimum reserve line. Automatic refill request timed out.\n'
x+= '[2026-03-19 06:35:21] [ERRO] PWR01 transient disturbed auxiliary pump control. Recovery completed with degraded margin.\n'
x+= '[2026-03-19 07:24:28] [ERRO] WTRPMP suction profile is inconsistent with expected coolant volume. Mechanical stress is increasing.\n'
x+= '[2026-03-19 06:05:32] [ERRO] FIRMWARE validation queue returned nonblocking fault set. Runtime proceeds in constrained mode.\n'
x+= '[2026-03-19 21:37:00] [CRIT] Final trip complete because WTANK07 remained under critical water level. FIRMWARE confirms safe shutdown state with all core operations halted.\n'
x+= '[2026-03-19 08:06:34] [ERRO] Heat transfer path to WSTPOOL2 is saturated. Dissipation lag continues to accumulate.\n'
x+= '[2026-03-19 06:01:09] [INFO] Coolant circulation pulse from WTRPMP is active. ECCS8 reports normal transfer demand.\n'

and after determining that verify works just weird (especially when extra logs are given) i decided to solve it without any LLM instead only using filtering
logs in S02E03_simple.py which works 
"""



# --- Download log ---
log_response = requests.get(LOG_URL, timeout=120)
log_response.raise_for_status()
RAW_LOG_PATH.write_text(log_response.text, encoding="utf-8")

raw_log_lines = RAW_LOG_PATH.read_text(encoding="utf-8").splitlines()
print(f"Downloaded log -> {RAW_LOG_PATH}")
print(f"Line count: {len(raw_log_lines)}")
print(f"Token estimate: {count_tokens(chr(10).join(raw_log_lines))}")

append_usage_log(
    AGENT_LOG_PATH, run_cost_summary, "download_failure_log",
    {"url": LOG_URL, "line_count": len(raw_log_lines), "size_bytes": len(log_response.content)},
    model=None,
)

# --- LLM analyzes log format (cached) ---
if SEVERITY_PROFILE_PATH.exists():
    severity_profile = json.loads(SEVERITY_PROFILE_PATH.read_text(encoding="utf-8"))
    print(f"\nLoaded cached severity profile from {SEVERITY_PROFILE_PATH}")
else:
    sample_lines = raw_log_lines[:FORMAT_SAMPLE_LINES]
    severity_profile = analyze_log_format(sample_lines)
    SEVERITY_PROFILE_PATH.write_text(
        json.dumps(severity_profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
print(f"Severity profile: {severity_profile}")

HIGH_SEVERITY_KEYWORDS: list[str] = ["crit", "error", "erro", "fatal", "fault", "fail", "trip", "shutdown", "emergency"]
LOW_SEVERITY_KEYWORDS: list[str] = ["warn"]
format_note: str = severity_profile["format_note"]

# Deduplicate raw log by message content (ignore timestamp) — removes repeated identical events
unique_log_lines: list[str] = deduplicate_logs_by_message(raw_log_lines)
print(f"Unique log lines after content deduplication: {len(unique_log_lines)} (was {len(raw_log_lines)})")

# Pre-compute severity-filtered sets once — avoids re-scanning raw log every iteration
high_severity_lines: list[str] = filter_lines_by_keywords(unique_log_lines, HIGH_SEVERITY_KEYWORDS)
all_severity_lines: list[str] = deduplicate_lines(
    high_severity_lines + filter_lines_by_keywords(unique_log_lines, LOW_SEVERITY_KEYWORDS)
)
print(f"High severity lines: {len(high_severity_lines)}, all severity (incl. warn): {len(all_severity_lines)}")

# --- Verify loop ---
feedback_payload: dict[str, Any] | None = None
feedback_focus_terms: set[str] = (
    set(json.loads(FOCUS_TERMS_PATH.read_text(encoding="utf-8")))
    if FOCUS_TERMS_PATH.exists()
    else set()
)
if feedback_focus_terms:
    print(f"Loaded focus terms from disk: {sorted(feedback_focus_terms)}")
iteration_history: list[dict[str, Any]] = []
previous_final_lines: list[str] = []
flag: str | None = None

for iteration in range(1, MAX_VERIFY_ITERATIONS + 1):
    print("\n" + "=" * 80)
    print(f"Iteration {iteration}/{MAX_VERIFY_ITERATIONS}")

    if iteration == 1:
        protected_lines = (
            filter_lines_by_keywords(unique_log_lines, [t.lower() for t in feedback_focus_terms])
            if feedback_focus_terms else []
        )
        reducible_lines = [l for l in high_severity_lines if l not in set(protected_lines)]
    else:
        currently_missing_terms = extract_feedback_focus_terms(feedback_payload) if feedback_payload else []
        protected_lines = filter_lines_by_keywords(
            unique_log_lines, [t.lower() for t in currently_missing_terms]
        )
        reducible_lines = [l for l in previous_final_lines if l not in set(protected_lines)]

    protected_lines = deduplicate_lines(protected_lines)
    print(f"Protected lines (focus terms, bypass LLM): {len(protected_lines)}, reducible: {len(reducible_lines)}")

    protected_token_count = count_tokens(build_submission_text(protected_lines))
    remaining_token_budget = TOKEN_LIMIT - protected_token_count

    if remaining_token_budget > 0 and reducible_lines:
        reduced_lines = call_openai_log_reducer(
            reducible_lines,
            focus_terms=feedback_focus_terms,
            feedback_payload=feedback_payload,
            format_note=format_note,
            target_token_limit=remaining_token_budget,
            action_name="llm_reduce_candidate_logs",
        )
    else:
        reduced_lines = []

    final_lines = trim_lines_to_token_limit(
        deduplicate_lines(protected_lines + reduced_lines), TOKEN_LIMIT, feedback_focus_terms
    )
    final_lines = sort_lines_chronologically(final_lines)
    final_text = build_submission_text(final_lines)
    final_token_count = count_tokens(final_text)
    FINAL_LOGS_PATH.write_text(final_text, encoding="utf-8")
    print(f"Submission lines: {len(final_lines)}, tokens: {final_token_count}")

    append_usage_log(
        AGENT_LOG_PATH, run_cost_summary, "prepare_submission",
        {
            "iteration": iteration,
            "protected_line_count": len(protected_lines),
            "reducible_line_count": len(reducible_lines),
            "submission_line_count": len(final_lines),
            "submission_token_count": final_token_count,
            "feedback_focus_terms": sorted(feedback_focus_terms),
        },
        model=None,
    )

    verification_response = verify_logs(final_text)
    FEEDBACK_PATH.write_text(
        json.dumps(verification_response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    append_usage_log(
        AGENT_LOG_PATH, run_cost_summary, "verify_submission",
        {"iteration": iteration, "response": verification_response},
        model=None,
    )
    print(json.dumps(verification_response, ensure_ascii=False, indent=2))

    flag = extract_flag(verification_response)
    iteration_history.append({
        "iteration": iteration,
        "protected_line_count": len(protected_lines),
        "reducible_line_count": len(reducible_lines),
        "submission_line_count": len(final_lines),
        "submission_token_count": final_token_count,
        "feedback_focus_terms": sorted(feedback_focus_terms),
        "response": verification_response,
        "flag": flag,
    })

    if flag:
        RESULT_PATH.write_text(flag, encoding="utf-8")
        print(f"Flag saved -> {RESULT_PATH}")
        break

    previous_final_lines = final_lines
    feedback_payload = verification_response
    new_terms = set(extract_feedback_focus_terms(verification_response))
    feedback_focus_terms |= new_terms
    FOCUS_TERMS_PATH.write_text(json.dumps(sorted(feedback_focus_terms), ensure_ascii=False), encoding="utf-8")
    print(f"Updated focus terms: {sorted(feedback_focus_terms)}")


# ------------------------------------------------------------------------------
# Phase 5: Persist summary
# ------------------------------------------------------------------------------

ITERATION_SUMMARY_PATH.write_text(
    json.dumps(
        {
            "run_id": RUN_ID,
            "openai_model": OPENAI_MODEL,
            "usage_summary": run_cost_summary,
            "severity_profile": severity_profile,
            "iterations": iteration_history,
            "flag": flag,
            "final_logs_path": str(FINAL_LOGS_PATH),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n" + "=" * 80)
if flag:
    print(f"Task solved. Flag: {flag}")
else:
    print("No flag obtained.")
    print(f"Inspect {FINAL_LOGS_PATH} and {FEEDBACK_PATH} for the latest attempt.")
import json
import os
import re
from pathlib import Path

import requests
import tiktoken
from dotenv import load_dotenv

load_dotenv()

AI_DEVS_API_KEY: str = os.environ["AI_DEVS_API_KEY"]
BASE_URL = "https://hub.ag3nts.org"
TASK = "failure"
LOG_URL = f"{BASE_URL}/data/{AI_DEVS_API_KEY}/failure.log"
VERIFY_URL = f"{BASE_URL}/verify"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_LOG_PATH = DATA_DIR / "failure.log"
TOKEN_LIMIT = 1400

enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def deduplicate_logs_by_message(lines: list[str]) -> list[str]:
    timestamp_pattern = re.compile(r'^\[?\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}(:\d{2})?\]?\s*')
    seen: set[str] = set()
    result = []
    for line in lines:
        message_key = timestamp_pattern.sub("", line).strip()
        if message_key not in seen:
            seen.add(message_key)
            result.append(line)
    return result


def filter_out_level(lines: list[str], level: str) -> list[str]:
    """Remove lines that contain the given log level marker."""
    pattern = re.compile(rf'\b{re.escape(level)}\b', re.IGNORECASE)
    return [line for line in lines if not pattern.search(line)]


def verify_logs(logs_text: str) -> dict:
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"logs": logs_text},
    }
    response = requests.post(VERIFY_URL, json=payload, timeout=60)
    response.raise_for_status() if response.status_code >= 500 else None
    return response.json()


# --- Download log (use cache if exists) ---
if RAW_LOG_PATH.exists():
    print(f"Using cached log: {RAW_LOG_PATH}")
else:
    print("Downloading log...")
    log_response = requests.get(LOG_URL, timeout=120)
    log_response.raise_for_status()
    RAW_LOG_PATH.write_text(log_response.text, encoding="utf-8")
    print(f"Downloaded -> {RAW_LOG_PATH}")

raw_lines = RAW_LOG_PATH.read_text(encoding="utf-8").splitlines()
print(f"Raw lines: {len(raw_lines)}, tokens: {count_tokens(chr(10).join(raw_lines))}")

# --- Step 1: deduplicate by message content ---
lines = deduplicate_logs_by_message(raw_lines)
print(f"After deduplication: {len(lines)} lines, tokens: {count_tokens(chr(10).join(lines))}")

# --- Step 2: drop INFO if over limit ---
if count_tokens(chr(10).join(lines)) > TOKEN_LIMIT:
    lines = filter_out_level(lines, "INFO")
    print(f"After removing INFO: {len(lines)} lines, tokens: {count_tokens(chr(10).join(lines))}")

# --- Step 3: drop WARN if still over limit ---
if count_tokens(chr(10).join(lines)) > TOKEN_LIMIT:
    lines = filter_out_level(lines, "WARN")
    print(f"After removing WARN: {len(lines)} lines, tokens: {count_tokens(chr(10).join(lines))}")

# --- Step 4: verify or fail ---
final_token_count = count_tokens(chr(10).join(lines))
if final_token_count > TOKEN_LIMIT:
    print(f"ERROR: still {final_token_count} tokens after all filtering — cannot fit in {TOKEN_LIMIT} token limit")
else:
    final_text = "\n".join(lines)
    print(f"\nSending {len(lines)} lines ({final_token_count} tokens)...")
    response = verify_logs(final_text)
    print(json.dumps(response, ensure_ascii=False, indent=2))

    flag_match = re.search(r'\{FLG:[^}]+\}', json.dumps(response))
    if flag_match:
        print(f"\nFlag: {flag_match.group(0)}")
        (DATA_DIR / "result.txt").write_text(flag_match.group(0), encoding="utf-8")

# ==============================================================================
# S01E05 - Railway - Aktywacja trasy X-01
# Faza 0: Konfiguracja i weryfikacja środowiska
# ==============================================================================

import os
import json
import time
import random
import datetime
import re
import copy
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Fail fast on missing env vars
AI_DEVS_API_KEY: str = os.environ["AI_DEVS_API_KEY"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

VERIFY_URL = "https://hub.ag3nts.org/verify"
TASK_NAME = "railway"

DATA_DIR = Path("tasks/S01E05/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = DATA_DIR / "agent_state.json"
LOG_FILE = DATA_DIR / "execution_log.jsonl"
HELP_FILE = DATA_DIR / "api_help_response.json"
PLAN_FILE = DATA_DIR / "execution_plan.json"
RESULT_FILE = DATA_DIR / "result.txt"

# Retry / backoff config
MAX_RETRIES_503 = 20
BACKOFF_BASE = 3.0      # seconds
BACKOFF_CAP = 90.0      # max single wait
JITTER_MAX = 2.0        # random jitter
MAX_FIX_ATTEMPTS = 3    # max LLM fix attempts per step

# ------------------------------------------------------------------------------
# Faza 1: Narzędzia - logging, state, HTTP client
# ------------------------------------------------------------------------------

import requests


def append_event_log(event: dict) -> None:
    """Append a structured event to the JSONL event log."""
    event["timestamp"] = datetime.datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_state() -> dict:
    """Load agent state from disk, or return initial state if not found."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "phase": "discover",
        "last_successful_action": None,
        "next_allowed_request_at": None,
        "attempt_count": 0,
        "error_count": 0,
        "execution_plan": [],
        "plan_step_index": 0,
        "fix_attempts": {},   # step_index -> count
        "flag_found": None,
    }


def save_state(state: dict) -> None:
    """Persist agent state to disk."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def extract_flag(data: dict | str) -> str | None:
    """Detect {FLG:...} pattern in any response data."""
    text = json.dumps(data) if isinstance(data, dict) else str(data)
    match = re.search(r'\{FLG:[^}]+\}', text)
    return match.group(0) if match else None


def _parse_rate_limit_headers(headers: dict) -> float | None:
    """
    Parse rate-limit headers and return Unix timestamp of when next
    request is allowed. Checks common header name variants.
    Returns None if no relevant headers found.
    """
    now = time.time()

    # Log all headers for debugging
    print(f"    Response headers: { {k: v for k, v in headers.items() if 'rate' in k.lower() or 'retry' in k.lower() or 'limit' in k.lower()} }")

    # Retry-After: value in seconds
    for key in ["Retry-After", "retry-after"]:
        val = headers.get(key)
        if val:
            try:
                return now + float(val)
            except ValueError:
                pass

    # X-RateLimit-Reset / RateLimit-Reset: Unix timestamp or seconds
    for key in ["X-RateLimit-Reset", "x-ratelimit-reset", "RateLimit-Reset", "ratelimit-reset"]:
        val = headers.get(key)
        if val:
            try:
                reset_val = float(val)
                # Looks like a Unix timestamp (> year 2000)
                if reset_val > 1_000_000_000:
                    return reset_val
                # Looks like seconds from now
                return now + reset_val
            except ValueError:
                pass

    # X-RateLimit-Reset-Requests (e.g. "30s" or "30")
    for key in ["X-RateLimit-Reset-Requests", "x-ratelimit-reset-requests"]:
        val = headers.get(key)
        if val:
            try:
                secs = float(str(val).rstrip("s"))
                return now + secs
            except ValueError:
                pass

    # If remaining is 0, set a default 60s safety buffer
    for key in ["X-RateLimit-Remaining", "x-ratelimit-remaining", "RateLimit-Remaining"]:
        val = headers.get(key)
        if val is not None:
            try:
                if int(val) == 0:
                    return now + 60.0
            except ValueError:
                pass

    return None


def send_verify_request(answer_payload: dict, state: dict) -> tuple[dict | None, dict]:
    """
    Send POST to /verify with retry on 503 and rate-limit handling.
    Updates state in-place. Returns (response_body | None, response_headers).
    """
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK_NAME,
        "answer": answer_payload,
    }

    for attempt in range(1, MAX_RETRIES_503 + 1):
        # Respect rate limit window from previous response
        if state.get("next_allowed_request_at"):
            wait_sec = state["next_allowed_request_at"] - time.time()
            if wait_sec > 0:
                print(f"  ⏳ Rate limit: waiting {wait_sec:.1f}s until reset...")
                append_event_log({"event": "rate_limit_wait", "seconds": round(wait_sec, 1)})
                time.sleep(wait_sec + 0.5)
            state["next_allowed_request_at"] = None

        state["attempt_count"] += 1
        action_name = answer_payload.get("action", "?")
        print(f"  → [{attempt}/{MAX_RETRIES_503}] action={action_name}")

        try:
            resp = requests.post(VERIFY_URL, json=payload, timeout=30)
        except requests.RequestException as e:
            append_event_log({"event": "request_error", "error": str(e)})
            print(f"  ✗ Request error: {e}")
            time.sleep(BACKOFF_BASE)
            continue

        # Always parse rate-limit headers after every response
        next_allowed = _parse_rate_limit_headers(dict(resp.headers))
        if next_allowed:
            state["next_allowed_request_at"] = next_allowed

        append_event_log({
            "event": "http_response",
            "action": action_name,
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text[:3000],
        })
        save_state(state)

        if resp.status_code == 503:
            backoff = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1))) + random.uniform(0, JITTER_MAX)
            state["error_count"] += 1
            print(f"  ⚠️  503 Service Unavailable — backoff {backoff:.1f}s (attempt {attempt})")
            time.sleep(backoff)
            continue

        if resp.status_code == 429:
            wait_sec = max(30.0, (state.get("next_allowed_request_at") or 0) - time.time())
            state["error_count"] += 1
            print(f"  🚫 429 Too Many Requests — waiting {wait_sec:.1f}s")
            time.sleep(wait_sec + 1.0)
            continue

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}

        print(f"  ✓ HTTP {resp.status_code}: {json.dumps(body, ensure_ascii=False)[:300]}")
        return body, dict(resp.headers)

    print(f"  ✗ Exhausted {MAX_RETRIES_503} attempts.")
    return None, {}


def save_api_doc(response_body: dict) -> None:
    """Save the help response to a local file for later reference."""
    with open(HELP_FILE, "w", encoding="utf-8") as f:
        json.dump(response_body, f, ensure_ascii=False, indent=2)
    print(f"  API documentation saved → {HELP_FILE}")


# Load or resume state
state = load_state()
print(f"\nAgent state: phase={state['phase']}, attempts={state['attempt_count']}, errors={state['error_count']}")

# ------------------------------------------------------------------------------
# Faza 2: Self-discovery API - wywołanie help i zapis dokumentacji lokalnie
# ------------------------------------------------------------------------------

if state["phase"] == "discover" or not HELP_FILE.exists():
    print("\n=== Phase 2: API Discovery (help) ===")
    help_response, _ = send_verify_request({"action": "help"}, state)

    if help_response is None:
        raise RuntimeError("Failed to get help response after all retries.")

    save_api_doc(help_response)
    append_event_log({"event": "api_help_saved", "keys": list(help_response.keys()) if isinstance(help_response, dict) else []})

    flag = extract_flag(help_response)
    if flag:
        print(f"\n🎉 FLAG FOUND IN HELP: {flag}")
        RESULT_FILE.write_text(flag, encoding="utf-8")
        state["flag_found"] = flag
        state["phase"] = "finish"
        save_state(state)
    else:
        state["phase"] = "plan"
        save_state(state)

# Always print cached docs for context
print("\n--- Cached API help response ---")
if HELP_FILE.exists():
    with open(HELP_FILE, "r", encoding="utf-8") as f:
        help_doc = json.load(f)
    print(json.dumps(help_doc, ensure_ascii=False, indent=2))
else:
    help_doc = {}

# ------------------------------------------------------------------------------
# Faza 3: LLM buduje deterministyczny plan wykonania na podstawie help
# ------------------------------------------------------------------------------

from openai import OpenAI

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def build_execution_plan_with_llm(help_doc: dict) -> list[dict]:
    """
    Ask LLM to interpret the API self-documentation and produce
    a step-by-step execution plan to activate route X-01.
    Returns a list of step dicts: [{step, action, params, description}].
    """
    prompt = f"""You are analyzing an undocumented railway control API.
Below is the complete response from the 'help' action (the API's self-documentation).

Your task: produce a minimal JSON execution plan to activate route X-01.

Rules:
1. Use ONLY action names and parameter names that appear verbatim in the docs.
2. Return a JSON object with key "steps" containing an array of step objects.
3. Each step must have: {{"step": N, "action": "exact_action_name", "params": {{...}}, "description": "why"}}
4. Include only the necessary steps in the correct order. Do NOT include 'help'.
5. If a step needs a value from a previous response (e.g. token, session ID),
   use placeholder format "{{{{FIELD_NAME}}}}" — e.g. {{"token": "{{{{token}}}}"}}
6. Do NOT guess or invent actions/params. Use exactly what the docs specify.

API documentation:
{json.dumps(help_doc, ensure_ascii=False, indent=2)}

Return ONLY the JSON object with "steps" key. No extra text."""

    print("  Asking LLM to build execution plan (gpt-4o-mini)...")
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    parsed = json.loads(raw)

    # Handle {"steps": [...]} or direct array or other wrapped forms
    if isinstance(parsed, list):
        return parsed
    for key in ("steps", "plan", "actions", "sequence"):
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    # Fallback: first list value
    return next((v for v in parsed.values() if isinstance(v, list)), [])


if state["phase"] == "plan" and help_doc:
    print("\n=== Phase 3: Building Execution Plan (LLM) ===")

    plan = build_execution_plan_with_llm(help_doc)

    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print(f"\nExecution plan ({len(plan)} steps):")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    state["execution_plan"] = plan
    state["plan_step_index"] = 0
    state["phase"] = "execute"
    save_state(state)

# If resuming from saved plan
if state["phase"] == "execute" and not state.get("execution_plan") and PLAN_FILE.exists():
    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        state["execution_plan"] = json.load(f)
    save_state(state)

# ------------------------------------------------------------------------------
# Faza 4: Egzekucja sekwencji akcji (state machine)
# ------------------------------------------------------------------------------


def resolve_placeholders(params: dict, previous_responses: list[dict]) -> dict:
    """
    Replace {{FIELD_NAME}} placeholders in params using data from previous
    API responses. Searches newest response first, including nested dicts.
    """
    resolved = copy.deepcopy(params)
    for key, value in resolved.items():
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            field_name = value[2:-2]
            for prev in reversed(previous_responses):
                if not isinstance(prev, dict):
                    continue
                # Direct match
                if field_name in prev:
                    resolved[key] = prev[field_name]
                    print(f"    Resolved {{{{{field_name}}}}} = {prev[field_name]}")
                    break
                # One level deep
                for nested in prev.values():
                    if isinstance(nested, dict) and field_name in nested:
                        resolved[key] = nested[field_name]
                        print(f"    Resolved (nested) {{{{{field_name}}}}} = {nested[field_name]}")
                        break
    return resolved


def ask_llm_to_fix_step(step: dict, error_response: dict, help_doc: dict, previous_responses: list) -> dict | None:
    """
    On validation error, ask LLM for a corrected step.
    Returns corrected step dict or None if unfixable.
    """
    prompt = f"""An API call returned an error. Suggest a corrected step.

Failed step:
{json.dumps(step, ensure_ascii=False, indent=2)}

Error response from API:
{json.dumps(error_response, ensure_ascii=False, indent=2)}

API documentation:
{json.dumps(help_doc, ensure_ascii=False, indent=2)}

Last 3 successful responses:
{json.dumps(previous_responses[-3:], ensure_ascii=False, indent=2)}

Return a JSON object: either the corrected step
  {{"step": N, "action": "...", "params": {{...}}, "description": "..."}}
or, if truly unfixable:
  {{"unfixable": true, "reason": "..."}}
"""
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(completion.choices[0].message.content)
    if result.get("unfixable"):
        print(f"  LLM: cannot fix — {result.get('reason')}")
        return None
    return result


def _is_api_error(body: dict) -> bool:
    """
    Determine if the response body represents a business-logic error
    (distinct from 503/429 which are handled at HTTP level).
    """
    if not isinstance(body, dict):
        return False
    # Explicit error fields
    if body.get("error"):
        return True
    # Numeric code indicating failure
    code = body.get("code")
    if code is not None:
        try:
            c = int(code)
            if c not in (0, 200):
                return True
        except (ValueError, TypeError):
            pass
    return False


if state["phase"] == "execute":
    print("\n=== Phase 4: Executing Action Sequence ===")

    plan: list[dict] = state.get("execution_plan", [])
    step_index: int = state.get("plan_step_index", 0)
    previous_responses: list[dict] = []

    if not plan:
        print("No execution plan loaded. Run Phase 3 first.")
    else:
        while step_index < len(plan) and not state.get("flag_found"):
            step = plan[step_index]
            step_key = str(step_index)
            fix_attempts = state.get("fix_attempts", {}).get(step_key, 0)

            print(f"\n--- Step {step_index + 1}/{len(plan)}: action={step.get('action')} ---")
            print(f"    {step.get('description', '')}")

            # Resolve dynamic placeholders from previous responses
            params = resolve_placeholders(step.get("params", {}), previous_responses)
            action_payload = {"action": step["action"], **params}

            response_body, _ = send_verify_request(action_payload, state)

            if response_body is None:
                print("  ✗ Step failed after all retries. Stopping.")
                save_state(state)
                break

            # Check for flag in any response
            flag = extract_flag(response_body)
            if flag:
                print(f"\n🎉 FLAG FOUND: {flag}")
                RESULT_FILE.write_text(flag, encoding="utf-8")
                append_event_log({"event": "flag_found", "flag": flag, "step": step_index})
                state["flag_found"] = flag
                state["phase"] = "finish"
                state["plan_step_index"] = step_index
                save_state(state)
                break

            # Handle business-logic errors
            if _is_api_error(response_body):
                print(f"  ⚠️  API error response: {json.dumps(response_body, ensure_ascii=False)[:300]}")
                append_event_log({"event": "api_error", "step": step_index, "response": response_body})

                if fix_attempts >= MAX_FIX_ATTEMPTS:
                    print(f"  ✗ Max fix attempts ({MAX_FIX_ATTEMPTS}) reached for step {step_index}. Stopping.")
                    save_state(state)
                    break

                # Ask LLM to fix the step
                fixed_step = ask_llm_to_fix_step(step, response_body, help_doc, previous_responses)
                if fixed_step:
                    print(f"  🔧 LLM fix applied: {json.dumps(fixed_step, ensure_ascii=False)}")
                    plan[step_index] = fixed_step
                    state["execution_plan"] = plan
                    state.setdefault("fix_attempts", {})[step_key] = fix_attempts + 1
                    append_event_log({"event": "step_fixed_by_llm", "step": step_index, "fixed": fixed_step})
                    save_state(state)
                    # Retry the fixed step (do not advance index)
                    continue
                else:
                    print("  ✗ LLM could not fix the step. Stopping.")
                    save_state(state)
                    break

            # Step succeeded — advance
            previous_responses.append(response_body)
            step_index += 1
            state["plan_step_index"] = step_index
            state["last_successful_action"] = step["action"]
            state.pop("fix_attempts", None)  # reset fix counter on success
            append_event_log({"event": "step_success", "step": step_index - 1, "action": step["action"]})
            save_state(state)

        if not state.get("flag_found") and step_index >= len(plan):
            print("\n⚠️  All plan steps executed but no flag detected.")
            print("Previous responses:")
            for i, r in enumerate(previous_responses):
                print(f"  [{i}] {json.dumps(r, ensure_ascii=False)[:300]}")

# ------------------------------------------------------------------------------
# Faza 5: Podsumowanie wykonania
# ------------------------------------------------------------------------------

print("\n=== Phase 5: Summary ===")

if LOG_FILE.exists():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    http_events = [e for e in events if e.get("event") == "http_response"]
    errors_503 = [e for e in http_events if e.get("status") == 503]
    errors_429 = [e for e in http_events if e.get("status") == 429]
    rate_waits = [e for e in events if e.get("event") == "rate_limit_wait"]
    total_wait = sum(e.get("seconds", 0) for e in rate_waits)

    print(f"  Total HTTP requests  : {len(http_events)}")
    print(f"  503 retries          : {len(errors_503)}")
    print(f"  429 retries          : {len(errors_429)}")
    print(f"  Rate limit waits     : {len(rate_waits)} ({total_wait:.1f}s total)")

print(f"  Total agent attempts : {state.get('attempt_count', 0)}")
print(f"  Error count          : {state.get('error_count', 0)}")
print(f"  Flag found           : {state.get('flag_found', 'NOT YET')}")

if state.get("flag_found"):
    print(f"\n✅ TASK COMPLETE — Flag: {state['flag_found']}")
    print(f"   Saved to: {RESULT_FILE}")

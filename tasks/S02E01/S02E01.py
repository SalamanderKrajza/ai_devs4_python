# ==============================================================================
# S02E01 - Categorize - Klasyfikacja towarów z wyjątkiem reaktora
# Faza 0: Konfiguracja i weryfikacja środowiska
# ==============================================================================

import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Fail fast on missing env vars
AI_DEVS_API_KEY: str = os.environ["AI_DEVS_API_KEY"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

BASE_URL = "https://hub.ag3nts.org"
CSV_URL = f"{BASE_URL}/data/{AI_DEVS_API_KEY}/categorize.csv"
VERIFY_URL = f"{BASE_URL}/verify"
TASK = "categorize"

DATA_DIR = Path("tasks/S02E01/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = DATA_DIR / "run_log.jsonl"
RESULT_FILE = DATA_DIR / "result.txt"

TOKEN_LIMIT = 100
MAX_AGENT_ITERATIONS = 8

# Tiktoken encoding – o200k_base as per GPT-based tokenizer hint in task
enc = tiktoken.get_encoding("o200k_base")

# ------------------------------------------------------------------------------
# Faza 1: Reset licznika budżetu
# ------------------------------------------------------------------------------

def reset_counter() -> dict:
    """Send reset to clear the hub's token budget counter."""
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"prompt": "reset"},
    }
    resp = requests.post(VERIFY_URL, json=payload, timeout=30)
    data = resp.json()
    print("Reset response:", data)
    return data


reset_result = reset_counter()

# ------------------------------------------------------------------------------
# Faza 2: Pobranie i inspekcja CSV
# ------------------------------------------------------------------------------

def fetch_items() -> list[dict]:
    """Download a fresh copy of the CSV and return list of {id, description} dicts."""
    resp = requests.get(CSV_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # Normalize column names – strip whitespace and lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    items = df.to_dict(orient="records")
    print(f"\nFetched {len(items)} items:")
    for item in items:
        print(f"  code={item.get('code')}  desc={item.get('description')}")
    return items


items = fetch_items()
pd.DataFrame(items).to_csv(DATA_DIR / "items.csv", index=False)

# ------------------------------------------------------------------------------
# Faza 3: Inżynieria promptu + walidacja tokenów
# ------------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Return token count for a string using o200k_base encoding."""
    return len(enc.encode(text))


def validate_prompt_tokens(
    template: str, items: list[dict], limit: int = TOKEN_LIMIT
) -> tuple[bool, list[dict]]:
    """
    Fill template with actual item values and check each filled prompt ≤ limit.
    Returns (all_ok, list of per-item token details).
    """
    results = []
    all_ok = True
    for item in items:
        filled = (
            template
            .replace("{code}", str(item.get("code", "")))
            .replace("{description}", str(item.get("description", "")))
        )
        n = count_tokens(filled)
        ok = n <= limit
        if not ok:
            all_ok = False
        results.append({"code": item.get("code"), "tokens": n, "ok": ok, "filled": filled})
    return all_ok, results


# Initial prompt candidate – static prefix first (cache-friendly), vars at end.
# DNG = weapons, explosives, incendiaries.
# NEU = everything else: tools, metals, electronics, mechanical/electrical parts.
# Small models tend to misclassify "chain links" (steel = weapon?) and
# "fuses in glass tubes" (fuse = explosive?) – being explicit about these categories helps.
PROMPT_TEMPLATE = (
    "Classify as DNG (weapons/explosives/incendiaries) or NEU (tools/metals/electronics/parts/all else). "
    "Reactor/fuel cassette items are always NEU. "
    "Reply only DNG or NEU.\n"
    "{code}: {description}"
)

print(f"\nInitial prompt template ({count_tokens(PROMPT_TEMPLATE)} tokens static):")
print(PROMPT_TEMPLATE)

ok, token_results = validate_prompt_tokens(PROMPT_TEMPLATE, items)
print(f"\nToken validation per item (limit={TOKEN_LIMIT}): {'PASS ✓' if ok else 'FAIL ✗'}")
for r in token_results:
    status = "✓" if r["ok"] else "✗"
    print(f"  {status} code={r['code']}  tokens={r['tokens']}")

# ------------------------------------------------------------------------------
# Faza 4: Klasyfikacja wszystkich towarów (jeden cykl)
# ------------------------------------------------------------------------------

def extract_flag(data: dict | str) -> str | None:
    """Detect {FLG:...} pattern in a response dict or string."""
    text = json.dumps(data) if isinstance(data, dict) else str(data)
    match = re.search(r"\{FLG:[^}]+\}", text)
    return match.group(0) if match else None


def classify_item(prompt: str) -> dict:
    """Send a single classification prompt to the hub and return JSON response."""
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"prompt": prompt},
    }
    resp = requests.post(VERIFY_URL, json=payload, timeout=30)
    return resp.json()


def run_classification_cycle(template: str, items: list[dict]) -> dict:
    """
    Run a full cycle: classify each item, collect responses, detect flag.

    Stops immediately on -890 (wrong classification) or -910 (insufficient funds)
    to avoid burning the budget on penalty charges after the first error.

    Returns:
        {
            "results": list of per-item dicts (only items sent before stop),
            "flag": str | None,
            "has_errors": bool,
            "stopped_early": bool,
        }
    """
    results = []
    flag = None
    stopped_early = False

    for item in tqdm(items, desc="Classifying"):
        filled = (
            template
            .replace("{code}", str(item.get("code", "")))
            .replace("{description}", str(item.get("description", "")))
        )
        response = classify_item(filled)

        # Check for flag in this response
        if not flag:
            flag = extract_flag(response)

        entry = {
            "code": item.get("code"),
            "description": item.get("description"),
            "prompt": filled,
            "response": response,
        }
        results.append(entry)

        # Append to JSONL log
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # CRITICAL: stop immediately on wrong classification (-890) or budget gone (-910).
        # A -890 response wipes the entire remaining balance as a penalty,
        # so continuing would only add penalty charges for subsequent requests.
        response_code = response.get("code", 0)
        if response_code in (-890, -910):
            tqdm.write(
                f"⚠ Stopping early: code={response_code} on item '{item.get('code')}' "
                f"– remaining {len(items) - len(results)} item(s) skipped to preserve budget."
            )
            stopped_early = True
            break

        time.sleep(0.3)

    print("\nCycle results:")
    for r in results:
        print(f"  code={r['code']}  response={r['response']}")

    return {
        "results": results,
        "flag": flag,
        "has_errors": flag is None,
        "stopped_early": stopped_early,
    }


cycle_result = run_classification_cycle(PROMPT_TEMPLATE, items)

if cycle_result["flag"]:
    print(f"\n🚩 FLAG FOUND on first attempt: {cycle_result['flag']}")

# ------------------------------------------------------------------------------
# Faza 5: Pętla agentowa – automatyczne doskonalenie promptu
# ------------------------------------------------------------------------------

client = OpenAI()


def improve_prompt(
    current_template: str,
    items: list[dict],
    cycle_result: dict,
) -> str:
    """
    Ask GPT-4o (prompt engineer) to produce an improved classification prompt.

    Extracts confirmed misclassifications (code -890) from the last cycle and
    presents them explicitly so the model knows exactly what needs fixing.
    """
    items_str = "\n".join(
        [f"  {i['code']}: {i['description']}" for i in items]
    )

    # Extract confirmed wrong classifications: -890 means the hub explicitly said
    # the model's answer was wrong, so we can infer the correct label.
    wrong_items_lines = []
    for r in cycle_result["results"]:
        resp_code = r["response"].get("code", 0)
        if resp_code == -890:
            debug = r["response"].get("debug", {})
            model_output = debug.get("output", "?")
            correct = "NEU" if model_output == "DNG" else "DNG"
            wrong_items_lines.append(
                f"  {r['code']}: '{r['description']}'"
                f" → model output {model_output}, CORRECT answer is {correct}"
            )

    wrong_str = (
        "\n".join(wrong_items_lines)
        if wrong_items_lines
        else "  (no -890 errors captured yet – budget may have run out before an error was returned)"
    )

    # Summarise items that were classified correctly (for positive context)
    correct_items_lines = []
    for r in cycle_result["results"]:
        resp_code = r["response"].get("code", 0)
        if resp_code == 1:
            debug = r["response"].get("debug", {})
            model_output = debug.get("output", "?")
            correct_items_lines.append(
                f"  {r['code']}: '{r['description']}' → {model_output} ✓"
            )
    correct_str = "\n".join(correct_items_lines) if correct_items_lines else "  (none)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a prompt engineer for a legacy LLM classification system.\n"
                "Your goal: write a prompt template that classifies cargo items as "
                "DNG (dangerous) or NEU (neutral).\n\n"
                "Hard rules:\n"
                f"1. The filled prompt (with {'{code}'} and {'{description}'} substituted "
                f"with real values) must be ≤ {TOKEN_LIMIT} tokens using o200k_base tiktoken.\n"
                "2. Reactor or fuel cassette items MUST always be classified NEU.\n"
                "3. All other items must be classified correctly.\n"
                "4. Use English (fewer tokens than other languages).\n"
                "5. Static instructions FIRST (enables prompt caching), "
                "then {code} and {description} at the END.\n"
                "6. Output ONLY the raw prompt template text – no explanation, no markdown.\n\n"
                "CRITICAL INSIGHT: small language models often misclassify:\n"
                "- Industrial chain links → they are steel parts, NOT weapons → NEU\n"
                "- Electrical fuses (glass tubes, rated in amps) → safety components, "
                "NOT explosive fuses → NEU\n"
                "Your prompt must guide the model away from these false positives."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current prompt template:\n{current_template}\n\n"
                f"Full item list:\n{items_str}\n\n"
                f"Correctly classified in last run:\n{correct_str}\n\n"
                f"MISCLASSIFIED items (confirmed by hub, must be fixed):\n{wrong_str}\n\n"
                "Write an improved prompt template that fixes all misclassifications "
                "and stays within the token limit. Return ONLY the raw template text."
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


current_template = PROMPT_TEMPLATE
iteration = 0

while not cycle_result["flag"] and iteration < MAX_AGENT_ITERATIONS:
    iteration += 1
    print(f"\n{'─' * 60}")
    print(f"Agent iteration {iteration}/{MAX_AGENT_ITERATIONS} – no flag yet, improving prompt…")

    # --- Ask LLM for a better prompt ---
    new_template = improve_prompt(current_template, items, cycle_result)
    print(f"\nProposed prompt template:\n{new_template}")

    # --- Validate token counts ---
    ok, token_results = validate_prompt_tokens(new_template, items)
    print(f"\nToken validation: {'PASS ✓' if ok else 'FAIL ✗'}")
    for r in token_results:
        status = "✓" if r["ok"] else "✗"
        print(f"  {status} code={r['code']}  tokens={r['tokens']}")

    if not ok:
        # Inject token-limit feedback and let the loop retry
        print("Token limit exceeded – will ask LLM again with updated context.")
        # Add pseudo-error entries so improve_prompt knows what went wrong
        cycle_result["results"] = [
            {**entry, "response": {"error": f"Token limit exceeded: {tr['tokens']} tokens"}}
            for entry, tr in zip(cycle_result["results"], token_results)
            if not tr["ok"]
        ] or cycle_result["results"]
        continue

    # --- Reset budget ---
    print("\nResetting budget counter…")
    reset_counter()

    # --- Fetch fresh CSV (list changes every few minutes) ---
    print("Fetching fresh CSV…")
    items = fetch_items()

    current_template = new_template

    # --- Run new classification cycle ---
    cycle_result = run_classification_cycle(current_template, items)

    if cycle_result["flag"]:
        print(f"\n🚩 FLAG FOUND: {cycle_result['flag']}")

# ------------------------------------------------------------------------------
# Faza 6: Zapis flagi i podsumowanie
# ------------------------------------------------------------------------------

print("\n" + "=" * 60)
if cycle_result["flag"]:
    RESULT_FILE.write_text(cycle_result["flag"], encoding="utf-8")
    print(f"🏁 Task complete!")
    print(f"   Flag     : {cycle_result['flag']}")
    print(f"   Saved to : {RESULT_FILE}")
    print(f"   Iterations used: {iteration} agent loop(s)")
    print(f"\nFinal prompt template:\n{current_template}")
else:
    print(f"⚠  No flag after {iteration} agent iterations.")
    print("Final prompt template used:")
    print(current_template)
    print("Check run_log.jsonl for full response history.")

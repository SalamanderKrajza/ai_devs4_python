# ==============================================================================
# S02E05 - drone - Przejęcie drona i zbombardowanie tamy
# Faza 0: Konfiguracja i weryfikacja środowiska
# ==============================================================================

import json
import os
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tasks.commons.llm_usage import (
    append_usage_log,
    calculate_usage_cost_usd,
    create_run_logs_dir,
    create_usage_summary,
    extract_openai_usage_metrics,
)
from tasks.commons.task_handler import AI_DEVS_API_KEY, send_verify

load_dotenv()

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

BASE_URL = "https://hub.ag3nts.org"
DRONE_DOCS_URL = f"{BASE_URL}/dane/drone.html"
DRONE_MAP_URL = f"{BASE_URL}/data/{AI_DEVS_API_KEY}/drone.png"
VERIFY_URL = f"{BASE_URL}/verify"
TASK = "drone"
PLANT_ID = "PWR6132PL"

VISION_MODEL = "gpt-5.4"
AGENT_MODEL = "gpt-4o"
MAX_AGENT_STEPS = 10
CHECKPOINT_EVERY = 3

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

run_dir, run_id = create_run_logs_dir(DATA_DIR, "s02e05")
log_path = run_dir / "llm_log.jsonl"
api_log_path = run_dir / "api_log.jsonl"
session_usage = create_usage_summary(AGENT_MODEL)

client = OpenAI()

print(f"Run ID: {run_id}")
print(f"Map URL: {DRONE_MAP_URL}")
print(f"Docs URL: {DRONE_DOCS_URL}")
print(f"Models: vision={VISION_MODEL}, agent={AGENT_MODEL}")

# ==============================================================================
# Faza 1: Pobranie i parsowanie dokumentacji API drona
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 1: Pobieranie dokumentacji API drona...")

docs_cache = DATA_DIR / "drone_docs.txt"
if docs_cache.exists():
    print(f"  Cache: {docs_cache}")
    docs_text = docs_cache.read_text(encoding="utf-8")
else:
    resp = requests.get(DRONE_DOCS_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    docs_text = soup.get_text(separator="\n", strip=True)
    docs_cache.write_text(docs_text, encoding="utf-8")
    print(f"  Pobrano -> {docs_cache}")

print(f"  Dokumentacja: {len(docs_text)} znaków")
print(f"\n--- Dokumentacja (pierwsze 2000 znakow) ---\n{docs_text[:2000]}\n---")

# ==============================================================================
# Faza 2: Analiza mapy (vision model)
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 2: Analiza mapy terenu (vision model)...")

map_analysis_cache = DATA_DIR / "map_analysis.json"
if map_analysis_cache.exists():
    print(f"  Cache: {map_analysis_cache}")
    map_analysis = json.loads(map_analysis_cache.read_text(encoding="utf-8"))
else:
    vision_response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at analyzing satellite/aerial map images with grid overlays. "
                    "You are meticulous about counting. When counting grid cells, you count "
                    "the number of DISTINCT CELLS (rectangles), not the number of grid lines."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a satellite map of a nuclear power plant area. "
                            "The map has a grid overlay dividing it into rectangular sectors.\n\n"
                            "STEP-BY-STEP INSTRUCTIONS:\n"
                            "1. Count the number of VERTICAL dividing lines inside the map (not the edges). "
                            "The number of columns = vertical_lines + 1.\n"
                            "2. Count the number of HORIZONTAL dividing lines inside the map (not the edges). "
                            "The number of rows = horizontal_lines + 1.\n"
                            "3. Locate the DAM — it is near an area of INTENSELY BLUE water "
                            "(the color was deliberately boosted to make it easy to find). "
                            "Identify which sector (column, row) it falls in.\n"
                            "4. Locate the POWER PLANT building and identify its sector.\n\n"
                            "Indexing: column 1 = leftmost, row 1 = topmost.\n\n"
                            "Think step by step, then return ONLY a JSON object:\n"
                            '{"vertical_lines": V, "horizontal_lines": H, '
                            '"cols": V+1, "rows": H+1, '
                            '"dam_col": X, "dam_row": Y, '
                            '"plant_col": PX, "plant_row": PY, '
                            '"reasoning": "detailed step-by-step explanation of your counting"}'
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": DRONE_MAP_URL, "detail": "high"},
                    },
                ],
            },
        ],
        temperature=0,
        max_completion_tokens=1000,
    )

    vision_text = vision_response.choices[0].message.content.strip()
    print(f"  Vision raw:\n{vision_text}")

    json_match = re.search(r"\{.*\}", vision_text, re.DOTALL)
    if json_match:
        map_analysis = json.loads(json_match.group(0))
    else:
        raise ValueError(f"Could not parse JSON from vision response: {vision_text}")

    map_analysis_cache.write_text(json.dumps(map_analysis, indent=2), encoding="utf-8")

    # Track vision cost
    v_metrics = extract_openai_usage_metrics(vision_response)
    v_cost = calculate_usage_cost_usd(VISION_MODEL, v_metrics)
    append_usage_log(
        log_path, session_usage, action="vision_map_analysis",
        model=VISION_MODEL, usage_metrics=v_metrics, cost_usd=v_cost,
        payload={"result": map_analysis},
    )
    print(f"  [vision] tokens={v_metrics['input_tokens']}/{v_metrics['output_tokens']} cost=${v_cost:.4f} | session=${session_usage['cost_usd']:.4f}")

print(f"\n  Analiza mapy: {json.dumps(map_analysis, indent=2)}")

dam_col = map_analysis["dam_col"]
dam_row = map_analysis["dam_row"]
print(f"  Sektor tamy: kolumna={dam_col}, wiersz={dam_row}")

# ==============================================================================
# Faza 3-4: Pętla agentowa z HITL
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 3-4: Pętla agentowa - programowanie drona")
print(f"  Cel deklarowany: elektrownia {PLANT_ID}")
print(f"  Cel faktyczny: tama w sektorze ({dam_col}, {dam_row})")
print(f"\n=== AGENT LOOP START ===")
print(f"Estimated steps: 3-5")
print(f"Checkpoint every: {CHECKPOINT_EVERY} actions")
print(f"Safety cap: {MAX_AGENT_STEPS} actions")

SYSTEM_PROMPT = f"""You are an expert drone programmer. Your task is to program a combat drone to complete a bombing mission.

MISSION BRIEFING:
- The declared target is the power plant with ID: {PLANT_ID}
- However, the ACTUAL bomb drop must happen at the DAM located in grid sector (column={dam_col}, row={dam_row})
- The map grid has {map_analysis.get('cols', '?')} columns and {map_analysis.get('rows', '?')} rows
- The drone carries one small-range explosive payload

DRONE API DOCUMENTATION:
{docs_text}

YOUR TASK:
Based on the documentation above, compose a sequence of drone instructions (as a JSON array of strings) that will:
1. Configure the drone for the mission
2. Set the target/destination appropriately
3. Arm and execute the mission so the bomb lands on the DAM sector

IMPORTANT RULES:
- The documentation may contain traps/conflicting function names - focus only on what's needed
- Read API error messages carefully and adjust
- If something goes wrong, consider using hardReset and starting fresh
- Keep instructions minimal - only what's needed for the mission

Return instructions as a JSON object:
{{"instructions": ["instruction1", "instruction2", ...]}}

If you receive an error from the API, analyze it and propose corrected instructions.
If you receive a flag ({{FLG:...}}), the mission is complete."""

history = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "Compose the drone instructions for the mission. "
            "Declare the power plant as target for official records, "
            f"but set the actual bombing sector to the dam at ({dam_col},{dam_row}). "
            "Return the instructions as JSON."
        ),
    },
]

flag = None

for step in range(1, MAX_AGENT_STEPS + 1):
    print(f"\n{'─' * 60}")
    print(f"Krok {step}/{MAX_AGENT_STEPS}")

    # --- LLM call ---
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=history,
        temperature=0,
        max_tokens=2000,
    )

    assistant_text = response.choices[0].message.content.strip()
    metrics = extract_openai_usage_metrics(response)
    cost = calculate_usage_cost_usd(AGENT_MODEL, metrics)
    append_usage_log(
        log_path, session_usage, action="agent_step",
        model=AGENT_MODEL, usage_metrics=metrics, cost_usd=cost,
        payload={"step": step, "response_preview": assistant_text[:300]},
    )
    print(f"  [step {step}] model={AGENT_MODEL} tokens={metrics['input_tokens']}/{metrics['output_tokens']} cost=${cost:.4f} | session=${session_usage['cost_usd']:.4f}")
    print(f"  [LLM] {assistant_text[:500]}")

    history.append({"role": "assistant", "content": assistant_text})

    # --- Extract instructions ---
    json_match = re.search(r'\{[^{}]*"instructions"\s*:\s*\[.*?\]\s*\}', assistant_text, re.DOTALL)
    if not json_match:
        array_match = re.search(r'\[.*?\]', assistant_text, re.DOTALL)
        if array_match:
            instructions = json.loads(array_match.group(0))
        else:
            print("  WARN: nie udalo sie wyciagnac instrukcji z odpowiedzi LLM")
            history.append({
                "role": "user",
                "content": 'Could not parse instructions. Return them as: {"instructions": ["...", "..."]}',
            })
            continue
    else:
        parsed = json.loads(json_match.group(0))
        instructions = parsed["instructions"]

    print(f"\n  Instrukcje ({len(instructions)}):")
    for i, instr in enumerate(instructions):
        print(f"    [{i}] {instr}")

    # --- Send to API ---
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"instructions": instructions},
    }

    print(f"\n  Wysylanie do {VERIFY_URL}...")
    try:
        api_resp = requests.post(VERIFY_URL, json=payload, timeout=60)
        api_result = api_resp.json()
    except Exception as e:
        api_result = {"error": str(e)}

    print(f"  API: {json.dumps(api_result, ensure_ascii=False)[:500]}")

    # Log API call
    from datetime import UTC, datetime
    with open(api_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step, "instructions": instructions, "api_response": api_result,
        }, ensure_ascii=False) + "\n")

    # --- Check for flag ---
    flag_match = re.search(r"\{FLG:[^}]+\}", json.dumps(api_result, ensure_ascii=False))
    if flag_match:
        flag = flag_match.group(0)
        print(f"\n  *** FLAGA ZNALEZIONA: {flag} ***")
        break

    # --- Feed error back to agent ---
    history.append({
        "role": "user",
        "content": (
            f"API response:\n{json.dumps(api_result, ensure_ascii=False, indent=2)}\n\n"
            "Analyze the error and provide corrected instructions. "
            "If configuration is corrupted, consider hardReset first. "
            'Return corrected instructions as JSON: {"instructions": [...]}'
        ),
    })

    # --- HITL Checkpoint ---
    if step % CHECKPOINT_EVERY == 0:
        print(f"\n{'=' * 60}")
        print(f"=== CHECKPOINT (krok {step}/{MAX_AGENT_STEPS}) ===")
        print(f"  Ostatnia odpowiedz API: {json.dumps(api_result, ensure_ascii=False)[:300]}")
        print(f"  Session: ${session_usage['cost_usd']:.4f} | tokens in={session_usage['input_tokens']} out={session_usage['output_tokens']}")
        print(f"  [Enter=kontynuuj | stop | lub wpisz hint]")
        user_input = input("  > ").strip()
        if user_input.lower() in ("stop", "abort"):
            print("  Zatrzymano przez usera.")
            break
        elif user_input and user_input.lower() not in ("k", "continue", ""):
            history.append({"role": "user", "content": f"User hint: {user_input}"})

# ==============================================================================
# Faza 5: Zapis wynikow i podsumowanie
# ==============================================================================

print(f"\n{'=' * 60}")
print(f"=== PODSUMOWANIE ===")
print(f"  Kroki: {step}")
print(f"  Tokeny: in={session_usage['input_tokens']} out={session_usage['output_tokens']}")
print(f"  Koszt sesji: ${session_usage['cost_usd']:.4f}")

if flag:
    result_path = DATA_DIR / "result.txt"
    result_path.write_text(flag, encoding="utf-8")
    print(f"  Flaga: {flag}")
    print(f"  Zapisano -> {result_path}")
else:
    print(f"  Brak flagi po {step} krokach.")
    print(f"  Logi: {run_dir}")

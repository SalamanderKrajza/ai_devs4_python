# ==============================================================================
# S03E03 - reactor - Agent nawigujący robotem przez reaktor
# Faza 0: Konfiguracja i importy
# ==============================================================================

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tasks.commons.llm_usage import (
    append_usage_log,
    calculate_usage_cost_usd,
    create_run_logs_dir,
    create_usage_summary,
    extract_gemini_usage_metrics,
    extract_openai_usage_metrics,
)
from tasks.commons.task_handler import AI_DEVS_API_KEY

load_dotenv()

TASK = "reactor"
VERIFY_URL = "https://hub.ag3nts.org/verify"
CHECKPOINT_EVERY = 10
MAX_STEPS = 120  # zabezpieczenie przed nieskończoną pętlą

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Faza 1: Wybór modelu
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 1: Wybór modelu")
print("  1 = gemini-2.5-flash  (szybki, tani)         [domyślny]")
print("  2 = gemini-2.5-pro    (mocniejszy)")
print("  3 = gpt-4o-mini       (OpenAI, tani)")
print("  4 = gpt-4o            (OpenAI, mocniejszy)")
choice = input("Wybierz [1-4, Enter=1]: ").strip() or "1"

MODEL_MAP = {
    "1": ("gemini", "gemini-2.5-flash"),
    "2": ("gemini", "gemini-2.5-pro"),
    "3": ("openai", "gpt-4o-mini"),
    "4": ("openai", "gpt-4o"),
}
PROVIDER, LLM_MODEL = MODEL_MAP.get(choice, ("gemini", "gemini-2.5-flash"))

run_dir, run_id = create_run_logs_dir(DATA_DIR, "s03e03")
llm_log_path   = run_dir / "llm_log.jsonl"
board_log_path = run_dir / "board_states.jsonl"
session_usage  = create_usage_summary(LLM_MODEL)

print(f"\nRun ID:    {run_id}")
print(f"Model:     {LLM_MODEL} ({PROVIDER})")
print(f"LLM log:   {llm_log_path}")
print(f"Board log: {board_log_path}")
print("Szacowane kroki: 10–30")

# ==============================================================================
# Faza 2: Inicjalizacja klienta LLM
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 2: Inicjalizacja klienta")

if PROVIDER == "gemini":
    from google import genai
    from google.genai import types as gtypes
    _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print("  Klient Gemini gotowy")
else:
    from openai import OpenAI
    _client = OpenAI()
    print("  Klient OpenAI gotowy")

# ==============================================================================
# Faza 3: Funkcje pomocnicze i definicja narzędzi
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 3: Definicja narzędzi")

_cmd_step = 0


def send_reactor_command(command: str) -> str:
    """Wyślij komendę do API reaktora i zaloguj stan planszy. Zwraca JSON string."""
    global _cmd_step
    _cmd_step += 1
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"command": command},
    }
    try:
        resp = requests.post(VERIFY_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except requests.HTTPError as e:
        result = {"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        result = {"error": str(e)}

    # Log stanu planszy (oddzielny plik od logów LLM)
    board_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "cmd_step": _cmd_step,
        "command": command,
        "response": result,
    }
    with open(board_log_path, "a", encoding="utf-8") as bf:
        bf.write(json.dumps(board_entry, ensure_ascii=False) + "\n")

    result_str = json.dumps(result, ensure_ascii=False)
    short = result_str[:500] + ("..." if len(result_str) > 500 else "")
    print(f"  [cmd={command!r} #{_cmd_step}] -> {short}")
    return result_str


def is_task_complete(board_json: str) -> bool:
    """Sprawdź czy odpowiedź API wskazuje na ukończenie zadania."""
    try:
        data = json.loads(board_json)
        if not isinstance(data, dict):
            return False
        # Sprawdź standardowy kod sukcesu
        if data.get("code") == 0:
            msg = str(data.get("message", ""))
            # Sukces tylko jeśli jest flaga lub komunikat o wygranej
            if any(k in msg for k in ["FLG:", "{{FLG", "congratulation", "success", "win",
                                       "gratulacje", "ukończono", "complete"]):
                return True
        return False
    except Exception:
        return False


SYSTEM_PROMPT = """You are an agent controlling a transport robot inside a nuclear reactor.

## Board layout (7 columns × 5 rows)
- Robot moves ONLY on row 5 (bottom row).
- Start: column 1, row 5 (marked P). Goal: column 7, row 5 (marked G).
- Reactor blocks (B) occupy exactly 2 vertical cells and move cyclically up/down.
- Blocks ONLY move when you issue a command — the clock is frozen otherwise.

## Available commands
send_command accepts exactly one of: start | right | left | wait | reset

## MANDATORY protocol

### Phase 1 — OBSERVATION (do this first, always)
1. Send "start".
2. Send "wait" at least 5 times in a row, observing carefully after each one.
3. For each block, track:
   - Which columns it occupies
   - Its current position in row 5 (yes/no)
   - Its movement direction (up or down)
4. After 5 waits write a brief summary of each block's movement pattern before moving.

### Phase 2 — NAVIGATION
Decide each move by predicting the state AFTER the command executes:
- RIGHT: safe only if row 5 in column (current+1) will be free after this move.
- WAIT:  stay in place if moving right is risky; observe one more step.
- LEFT:  retreat if your current column is also becoming dangerous.
- RESET: use this if the API response says the robot was crushed/destroyed.

Key rule: after every command ALL blocks move one step. Always predict the next
state before committing. Do not move right into a block that is one step away
from row 5 in that column.

## Success
When the robot reaches column 7, row 5 (G), the task is complete. Report success."""

# --- Gemini tool definition ---
GEMINI_TOOL = None
if PROVIDER == "gemini":
    GEMINI_TOOL = gtypes.Tool(function_declarations=[
        gtypes.FunctionDeclaration(
            name="send_command",
            description=(
                "Send one reactor command and receive the updated board state. "
                "Valid commands: start | right | left | wait | reset"
            ),
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "command": gtypes.Schema(
                        type=gtypes.Type.STRING,
                        description="One of: start, right, left, wait, reset",
                    )
                },
                required=["command"],
            ),
        )
    ])

# --- OpenAI tool definition ---
OPENAI_TOOLS = None
if PROVIDER == "openai":
    OPENAI_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "send_command",
                "description": (
                    "Send one reactor command and receive the updated board state. "
                    "Valid commands: start | right | left | wait | reset"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["start", "right", "left", "wait", "reset"],
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]

print(f"  Narzędzie send_command gotowe ({PROVIDER})")

# ==============================================================================
# Faza 4: Pętla agentowa (HITL)
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 4: Pętla agentowa")
print(f"  Checkpoint co {CHECKPOINT_EVERY} kroków LLM | k/Enter=kontynuuj, stop=przerwij, inne=hint")
print()

step = 0
stop_requested = False
task_done = False

# ─────────────────────────────────────────────────────────
# GEMINI loop
# ─────────────────────────────────────────────────────────
if PROVIDER == "gemini":
    history = [
        gtypes.Content(
            role="user",
            parts=[gtypes.Part(text="Execute the reactor navigation task. Follow the mandatory protocol.")]
        )
    ]

    while not stop_requested and not task_done and step < MAX_STEPS:
        step += 1

        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=history,
            config=gtypes.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[GEMINI_TOOL],
                temperature=0,
            ),
        )

        metrics = extract_gemini_usage_metrics(response)
        cost = calculate_usage_cost_usd(LLM_MODEL, metrics)
        append_usage_log(
            llm_log_path, session_usage,
            action="agent_step", model=LLM_MODEL,
            usage_metrics=metrics, cost_usd=cost,
            payload={"step": step},
        )
        print(
            f"[step {step}] tokens={metrics['input_tokens']}/{metrics['output_tokens']} "
            f"cost=${cost:.4f} | session=${session_usage['cost_usd']:.4f}"
        )

        model_content = response.candidates[0].content
        history.append(model_content)

        fn_calls = [
            p.function_call
            for p in model_content.parts
            if hasattr(p, "function_call") and p.function_call is not None
        ]
        texts = [
            p.text for p in model_content.parts
            if hasattr(p, "text") and p.text
        ]

        if texts:
            print(f"\n[agent] {''.join(texts)}")

        if fn_calls:
            fn_parts = []
            for fc in fn_calls:
                args = dict(fc.args) if fc.args else {}
                command = args.get("command", "wait")
                board_json = send_reactor_command(command)
                if is_task_complete(board_json):
                    task_done = True
                fn_parts.append(
                    gtypes.Part.from_function_response(
                        name="send_command",
                        response={"result": board_json},
                    )
                )
            history.append(gtypes.Content(role="user", parts=fn_parts))
            if task_done:
                print("\n  ZADANIE UKOŃCZONE!")
                break
        elif not texts:
            print("\n[WARN] Agent nie zwrócił ani tekstu ani tool call.")

        # Checkpoint co CHECKPOINT_EVERY kroków lub gdy agent pisze tekst bez tool call
        needs_checkpoint = (step % CHECKPOINT_EVERY == 0) or (texts and not fn_calls)
        if not task_done and needs_checkpoint:
            print("\n--- CHECKPOINT ---")
            print(f"  Krok LLM: {step} | Komend do API: {_cmd_step} | Koszt: ${session_usage['cost_usd']:.4f}")
            user_in = input("  Kontynuuj? [k/Enter=tak, stop=przerwij, inne=hint]: ").strip()
            if user_in.lower() in ("stop", "abort"):
                stop_requested = True
            elif user_in and user_in.lower() not in ("k", "continue", ""):
                history.append(gtypes.Content(role="user", parts=[gtypes.Part(text=user_in)]))
            elif texts and not fn_calls:
                # Agent zatrzymał się na tekście — pchnij go dalej
                history.append(gtypes.Content(role="user", parts=[gtypes.Part(text="Continue with the task.")]))

# ─────────────────────────────────────────────────────────
# OPENAI loop
# ─────────────────────────────────────────────────────────
else:
    messages = [
        {"role": "user", "content": "Execute the reactor navigation task. Follow the mandatory protocol."}
    ]

    while not stop_requested and not task_done and step < MAX_STEPS:
        step += 1

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        metrics = extract_openai_usage_metrics(response)
        cost = calculate_usage_cost_usd(LLM_MODEL, metrics)
        append_usage_log(
            llm_log_path, session_usage,
            action="agent_step", model=LLM_MODEL,
            usage_metrics=metrics, cost_usd=cost,
            payload={"step": step},
        )
        print(
            f"[step {step}] tokens={metrics['input_tokens']}/{metrics['output_tokens']} "
            f"cost=${cost:.4f} | session=${session_usage['cost_usd']:.4f}"
        )

        oai_msg = response.choices[0].message
        tool_calls = oai_msg.tool_calls or []
        text_content = oai_msg.content or ""

        # Skonwertuj message do dict (bezpieczne do ponownego wysyłania do API)
        msg_dict: dict = {"role": "assistant", "content": text_content}
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(msg_dict)

        if text_content:
            print(f"\n[agent] {text_content}")

        if tool_calls:
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                command = args.get("command", "wait")
                board_json = send_reactor_command(command)
                if is_task_complete(board_json):
                    task_done = True
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": board_json,
                })
            if task_done:
                print("\n  ZADANIE UKOŃCZONE!")
                break
        elif not text_content:
            print("\n[WARN] Agent nie zwrócił ani tekstu ani tool call.")

        # Checkpoint
        needs_checkpoint = (step % CHECKPOINT_EVERY == 0) or (text_content and not tool_calls)
        if not task_done and needs_checkpoint:
            print("\n--- CHECKPOINT ---")
            print(f"  Krok LLM: {step} | Komend do API: {_cmd_step} | Koszt: ${session_usage['cost_usd']:.4f}")
            user_in = input("  Kontynuuj? [k/Enter=tak, stop=przerwij, inne=hint]: ").strip()
            if user_in.lower() in ("stop", "abort"):
                stop_requested = True
            elif user_in and user_in.lower() not in ("k", "continue", ""):
                messages.append({"role": "user", "content": user_in})
            elif text_content and not tool_calls:
                messages.append({"role": "user", "content": "Continue with the task."})

if step >= MAX_STEPS and not task_done:
    print(f"\n[WARN] Osiągnięto maksymalną liczbę kroków ({MAX_STEPS}).")

# ==============================================================================
# Faza 5: Podsumowanie
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 5: Podsumowanie")
print(f"  Status:        {'UKOŃCZONO' if task_done else 'PRZERWANO / W TOKU'}")
print(f"  Kroki LLM:     {step}")
print(f"  Komend do API: {_cmd_step}")
print(f"  Tokeny:        in={session_usage['input_tokens']} out={session_usage['output_tokens']}")
print(f"  Koszt:         ${session_usage['cost_usd']:.4f}")
print(f"  LLM log:       {llm_log_path}")
print(f"  Board log:     {board_log_path}")

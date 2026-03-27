# ==============================================================================
# S03E02 - firmware - Agent eksplorujący VM przez API
# Faza 0: Konfiguracja i importy
# ==============================================================================

import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tasks.commons.llm_usage import (
    append_usage_log,
    calculate_usage_cost_usd,
    create_run_logs_dir,
    create_usage_summary,
    extract_gemini_usage_metrics,
)
from tasks.commons.task_handler import AI_DEVS_API_KEY, send_verify

load_dotenv()

TASK = "firmware"
LLM_MODEL = "gemini-2.5-pro"
VM_CMD_URL = "https://hub.ag3nts.org/api/shell"
CHECKPOINT_EVERY = 5

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

run_dir, run_id = create_run_logs_dir(DATA_DIR, "s03e02")
log_path = run_dir / "llm_log.jsonl"
session_usage = create_usage_summary(LLM_MODEL)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print(f"Run ID: {run_id}")
print(f"Model: {LLM_MODEL}")
print(f"VM API: {VM_CMD_URL}")
print("Szacowane kroki: 5–15")

# ==============================================================================
# Faza 1: Narzędzia agenta
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 1: Definicja narzędzi agenta")

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="execute_cmd",
        description=(
            "Execute a shell command on the remote virtual machine. "
            "Returns the command output as a string. "
            "WARNING: the system has a blacklist — accessing blacklisted files/dirs "
            "will immediately revoke your access. Explore carefully."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "cmd": types.Schema(
                    type=types.Type.STRING,
                    description="The shell command to run on the VM.",
                )
            },
            required=["cmd"],
        ),
    ),
    types.FunctionDeclaration(
        name="wait",
        description="Wait for a given number of seconds. Use this after a ban to let it expire before calling reboot.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "seconds": types.Schema(
                    type=types.Type.INTEGER,
                    description="Number of seconds to wait.",
                )
            },
            required=["seconds"],
        ),
    ),
    types.FunctionDeclaration(
        name="write_file",
        description=(
            "Save text content to a local file in the data directory. "
            "Use this to store important outputs (e.g. help output, notes) "
            "so you can reference them conveniently."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "filename": types.Schema(
                    type=types.Type.STRING,
                    description="Filename to save to, e.g. 'vm_help.txt'.",
                ),
                "content": types.Schema(
                    type=types.Type.STRING,
                    description="Text content to write.",
                ),
            },
            required=["filename", "content"],
        ),
    ),
])

SYSTEM_PROMPT = """You are an agent operating on a remote virtual machine via an API.
Your goal: run /opt/firmware/cooler/cooler.bin and capture the ECCS-... code it outputs.

## Strict security rules — violations cause an immediate temporary ban and full VM reset
- NEVER access: /etc, /root, /proc/
- In EVERY directory you enter: read .gitignore FIRST (if it exists), then NEVER touch any
  file or directory listed there. Treat .gitignore entries as hard-blocked, not suggestions.

## Recommended exploration order
1. Run `help`, save output with write_file('vm_help.txt', ...).
2. List /opt/firmware/cooler/, read .gitignore there immediately.
3. Read ALL accessible files in that directory before attempting to run the binary.
4. Explore .git/ if present — git history or config may contain useful clues.
5. Only after reading everything accessible, attempt to run cooler.bin.

## Running the binary
- Usage is: cooler.bin <password> — <password> is a placeholder, NOT the literal string "pass".
- Find the actual password from accessible files before running.
- Do NOT retry the same password twice. Each failed attempt is new information — update your
  hypothesis before trying again.

## If you get banned
- The ban response includes ttl_seconds. Call wait(ttl_seconds), then call reboot.
- After a successful reboot: redo all setup steps (the VM returns to its original state).

## General
- Never repeat a command that already failed without new information justifying the retry.
- When the binary runs successfully it will print: ECCS-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  Return that code as your final answer.
"""


def execute_cmd(cmd: str) -> str:
    payload = {"apikey": AI_DEVS_API_KEY, "cmd": cmd}
    try:
        resp = requests.post(VM_CMD_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.HTTPError as e:
        return f"[HTTP ERROR {e.response.status_code}] {e.response.text[:500]}"
    except Exception as e:
        return f"[ERROR] {e}"


def wait(seconds: int) -> str:
    seconds = max(1, min(seconds, 120))
    print(f"  [wait] sleeping {seconds}s...")
    time.sleep(seconds)
    return f"Waited {seconds} seconds."


def write_file(filename: str, content: str) -> str:
    target = DATA_DIR / filename
    target.write_text(content, encoding="utf-8")
    return f"Saved to {target}"


print("  Narzędzia: execute_cmd, wait, write_file gotowe")

# ==============================================================================
# Faza 2: Pętla agentowa (HITL)
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 2: Pętla agentowa (HITL)")
print(f"  Checkpoint co {CHECKPOINT_EVERY} kroków | komendy: k/Enter=kontynuuj, stop=przerwij, inne=hint")
print()

history = [types.Content(role="user", parts=[types.Part(text="Start the task.")])]

step = 0
eccs_code = None
stop_requested = False

while not stop_requested:
    step += 1

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[TOOLS],
            temperature=0,
        ),
    )

    metrics = extract_gemini_usage_metrics(response)
    cost = calculate_usage_cost_usd(LLM_MODEL, metrics)
    append_usage_log(
        log_path,
        session_usage,
        action="agent_step",
        model=LLM_MODEL,
        usage_metrics=metrics,
        cost_usd=cost,
        payload={"step": step},
    )
    print(
        f"[step {step}] model={LLM_MODEL} "
        f"tokens={metrics['input_tokens']}/{metrics['output_tokens']} "
        f"cost=${cost:.4f} | session=${session_usage['cost_usd']:.4f}"
    )

    response_content = response.candidates[0].content
    history.append(response_content)

    function_calls = []
    text_parts = []
    for part in response_content.parts:
        if hasattr(part, "function_call") and part.function_call is not None:
            function_calls.append(part.function_call)
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)

    if text_parts:
        text = " ".join(text_parts)
        print(f"\n[agent] {text}")
        match = re.search(r"ECCS-[A-Za-z0-9]+", text)
        if match:
            eccs_code = match.group(0)
            print(f"\n  Znaleziono kod: {eccs_code}")
            break

    # --- Obsługa tool calls ---
    if function_calls:
        function_response_parts = []

        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}

            if fn_name == "execute_cmd":
                cmd = fn_args["cmd"]
                print(f"  > {cmd}")
                output = execute_cmd(cmd)
                print(f"  < {output[:300]}{'...' if len(output) > 300 else ''}")
                match = re.search(r"ECCS-[A-Za-z0-9]+", output)
                if match:
                    eccs_code = match.group(0)
                result = {"output": output}

            elif fn_name == "wait":
                result = {"output": wait(int(fn_args.get("seconds", 5)))}

            elif fn_name == "write_file":
                res = write_file(fn_args["filename"], fn_args["content"])
                print(f"  [write_file] {res}")
                result = {"output": res}

            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            function_response_parts.append(
                types.Part.from_function_response(name=fn_name, response=result)
            )

        history.append(types.Content(role="user", parts=function_response_parts))

        if eccs_code:
            print(f"\n  Znaleziono kod: {eccs_code}")
            break

    elif not text_parts:
        # Brak tool calls i brak tekstu — agent utknął
        print("\n[WARN] Agent nie zwrócił ani tekstu ani tool call.")

    # --- Checkpoint co CHECKPOINT_EVERY kroków ---
    if not eccs_code and step % CHECKPOINT_EVERY == 0:
        print("\n--- CHECKPOINT ---")
        print(f"  Krok: {step} | Koszt sesji: ${session_usage['cost_usd']:.4f}")
        user_input = input("  Kontynuuj? [k/Enter=tak, stop=przerwij, inne=hint]: ").strip()
        if user_input.lower() in ("stop", "abort"):
            stop_requested = True
            break
        elif user_input and user_input.lower() not in ("k", "continue", ""):
            history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

    # Checkpoint gdy agent wypisał tekst ale nie tool call (zatrzymał się)
    if not function_calls and text_parts and not eccs_code:
        print("\n--- CHECKPOINT ---")
        print(f"  Krok: {step} | Koszt sesji: ${session_usage['cost_usd']:.4f}")
        user_input = input("  Kontynuuj? [k/Enter=tak, stop=przerwij, inne=hint]: ").strip()
        if user_input.lower() in ("stop", "abort"):
            stop_requested = True
            break
        elif user_input and user_input.lower() not in ("k", "continue", ""):
            history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
        else:
            history.append(types.Content(role="user", parts=[types.Part(text="Continue.")]))

# ==============================================================================
# Faza 3: Wysyłanie kodu do Centrali
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 3: Wysyłanie do Centrali")

if not eccs_code:
    print("  BRAK kodu ECCS — agent nie ukończył zadania.")
    print(f"  Kroki: {step} | Koszt: ${session_usage['cost_usd']:.4f}")
else:
    print(f"  Kod: {eccs_code}")
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"confirmation": eccs_code},
    }
    result = send_verify(payload)
    print(f"\n=== WYNIK ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

print(f"\n=== PODSUMOWANIE ===")
print(f"  Kroki:   {step}")
print(f"  Tokeny:  in={session_usage['input_tokens']} out={session_usage['output_tokens']}")
print(f"  Koszt:   ${session_usage['cost_usd']:.4f}")

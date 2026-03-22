# ==============================================================================
# S02E04 - mailbox - Agentowa pętla z dynamicznym function calling
# Faza 0: Konfiguracja i weryfikacja środowiska
# ==============================================================================

import json
import os
import re
import sys
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

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

ZMAIL_URL = "https://hub.ag3nts.org/api/zmail"
TASK = "mailbox"
MODEL = "gemini-3-flash-preview"
MAX_STEPS = 40

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

run_dir, run_id = create_run_logs_dir(DATA_DIR, "s02e04")
log_path = run_dir / "llm_log.jsonl"
api_log_path = run_dir / "api_log.jsonl"
usage = create_usage_summary(MODEL)
print(f"Run ID: {run_id}")
print(f"Logi API: {api_log_path}")

# ==============================================================================
# Faza 1: Implementacje narzędzi API
# ==============================================================================

def call_zmail(**kwargs) -> dict:
    """Wywołuje API skrzynki mailowej. Nigdy nie rzuca wyjątku - błędy zwracane jako dict."""
    from datetime import UTC, datetime

    payload = {"apikey": AI_DEVS_API_KEY, **kwargs}
    resp = requests.post(ZMAIL_URL, json=payload, timeout=30)
    try:
        result = resp.json()
    except Exception:
        result = {"ok": False, "error": resp.text, "status_code": resp.status_code}
    if not resp.ok and "error" not in result:
        result["error"] = f"HTTP {resp.status_code}"
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "func": "call_zmail",
        "args": kwargs,
        "result": result,
    }
    with open(api_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return result


def submit_answer(password: str, date: str, confirmation_code: str) -> dict:
    """Wysyła trzy znalezione wartości do huba weryfikacyjnego."""
    from datetime import UTC, datetime

    answer = {"password": password, "date": date, "confirmation_code": confirmation_code}
    try:
        result = send_verify({"apikey": AI_DEVS_API_KEY, "task": TASK, "answer": answer})
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "func": "submit_answer",
        "args": answer,
        "result": result,
    }
    with open(api_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return result

# ==============================================================================
# Faza 2: Parsowanie helpa i dynamiczne budowanie deklaracji narzędzi
# ==============================================================================

def _infer_schema(description: str) -> dict:
    """Wyprowadź JSON Schema type z opisu parametru z helpa."""
    dl = description.lower()
    if "array" in dl:
        # np. "Numeric rowID, 32-char messageID, or an array of them."
        return {"type": "array", "items": {"type": "string"}}
    if "integer" in dl or (
        "numeric" in dl and "messageid" not in dl and "32-char" not in dl
    ):
        # np. "Required. Numeric thread identifier." lub "Optional. Integer >= 1."
        return {"type": "integer"}
    return {"type": "string"}


def build_zmail_declarations(help_response: dict) -> list:
    """
    Buduje FunctionDeclaration dla każdej akcji z odpowiedzi helpa.
    Każda akcja dostaje nazwę 'zmail_<action>' i właściwe typy parametrów.
    Akcja 'help' jest pomijana - została już wywołana.
    """
    declarations = []
    for action_name, action_info in help_response.get("actions", {}).items():
        if action_name == "help":
            continue

        params_raw = action_info.get("params", {})
        properties: dict = {}
        required_params: list = []

        if isinstance(params_raw, dict):
            for param_name, param_desc in params_raw.items():
                if param_name == "action":
                    continue  # pomijamy - jest zakodowane w nazwie funkcji
                schema = _infer_schema(str(param_desc))
                schema["description"] = str(param_desc)
                properties[param_name] = schema
                if str(param_desc).strip().lower().startswith("required"):
                    required_params.append(param_name)

        declarations.append(
            types.FunctionDeclaration(
                name=f"zmail_{action_name}",
                description=action_info.get("description", action_name),
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            )
        )
    return declarations

# ==============================================================================
# Faza 3: Odczytanie helpa i budowanie zestawu narzędzi agenta
# ==============================================================================

print("\nPobieranie helpa API zmail...")
help_response = call_zmail(action="help")
zmail_declarations = build_zmail_declarations(help_response)
print(f"Zbudowano {len(zmail_declarations)} narzędzi zmail: {[d.name for d in zmail_declarations]}")

SUBMIT_DECLARATION = types.FunctionDeclaration(
    name="submit_answer",
    description=(
        "Wysyła zebrane wartości do huba weryfikacyjnego. "
        "Wywołaj gdy masz wszystkie trzy: password, date (YYYY-MM-DD), confirmation_code (SEC- + 32 znaki). "
        "Hub zwróci feedback - jeśli coś jest błędne, kontynuuj szukanie."
    ),
    parameters={
        "type": "object",
        "properties": {
            "password": {"type": "string", "description": "Hasło do systemu pracowniczego"},
            "date": {"type": "string", "description": "Data planowanego ataku, format YYYY-MM-DD"},
            "confirmation_code": {
                "type": "string",
                "description": "Kod potwierdzenia SEC- + 32 znaki (36 znaków łącznie)",
            },
        },
        "required": ["password", "date", "confirmation_code"],
    },
)

tools = types.Tool(function_declarations=[*zmail_declarations, SUBMIT_DECLARATION])

# ==============================================================================
# Faza 4: Prompt systemowy
# ==============================================================================

SYSTEM_PROMPT = """Jesteś agentem przeszukującym skrzynkę mailową operatora Systemu.
Szukasz trzech wartości:
1. date - data planowanego ataku na elektrownię (format YYYY-MM-DD)
2. password - hasło do systemu pracowniczego
3. confirmation_code - kod potwierdzenia format SEC- + 32 znaki (36 znaków łącznie)

Co wiesz:
- Wiktor z ruchu oporu wysłał donos z domeny proton.me
- Skrzynka jest aktywna - nowe maile mogą napływać w trakcie pracy
- Dział bezpieczeństwa wysłał ticket z kodem potwierdzenia
- Hasło do systemu pracowniczego prawdopodobnie jest w jakimś mailu na tej skrzynce

Zasady pracy:
1. ZAWSZE pobieraj pełną treść maila przed wyciąganiem wniosków
2. Śledź stan znalezionych wartości: date=?, password=?, confirmation_code=?
3. Raz znalezionej wartości NIE szukaj ponownie — skieruj uwagę na brakujące
4. Jeśli wartość NIEZNANA i nic nie znajdziesz — skrzynka jest aktywna, spróbuj ponownie
5. Jeśli widzisz korektę wartości (np. "poprawny kod to...") — zawsze używaj najnowszej
6. Gdy masz WSZYSTKIE 3 wartości — NATYCHMIAST wywołaj submit_answer
7. Jeśli hub zwróci błąd dla konkretnej wartości — szukaj ponownie TYLKO tej wartości"""

# ==============================================================================
# Faza 5: Pętla agentowa
# ==============================================================================

client = genai.Client(api_key=GEMINI_API_KEY)

history: list = []
flag = None

initial_message = (
    "Zacznij przeszukiwanie skrzynki mailowej. "
    "Zdobądź date (data ataku), password (hasło pracownicze) i confirmation_code (kod SEC-)."
)

history.append(types.Content(role="user", parts=[types.Part(text=initial_message)]))
print(f"\n[START] {initial_message}\n")

for step in range(MAX_STEPS):
    print(f"{'─' * 60}")
    print(f"Krok {step + 1}/{MAX_STEPS}")

    response = client.models.generate_content(
        model=MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tools],
            temperature=0,
        ),
    )

    usage_metrics = extract_gemini_usage_metrics(response)
    cost = calculate_usage_cost_usd(MODEL, usage_metrics)

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
        print(f"[AGENT] {' '.join(text_parts)}")

    if not function_calls:
        print("Agent nie wywołał żadnego narzędzia - kończę pętlę.")
        break

    function_response_parts = []
    step_calls = []

    for fc in function_calls:
        fn_name = fc.name
        fn_args = dict(fc.args) if fc.args else {}
        print(f"  → {fn_name}({fn_args})")

        if fn_name.startswith("zmail_"):
            # Nazwa funkcji = 'zmail_<action>' → wyciągamy action i wywołujemy
            action = fn_name[len("zmail_"):]
            result = call_zmail(action=action, **fn_args)
        elif fn_name == "submit_answer":
            result = submit_answer(**fn_args)
            append_usage_log(
                log_path, usage,
                action="submit_answer",
                payload={"args": fn_args, "result": result},
            )
            flag_match = re.search(r"\{FLG:[^}]+\}", json.dumps(result))
            if flag_match:
                flag = flag_match.group(0)
                print(f"\n  *** FLAGA: {flag} ***")
        else:
            result = {"error": f"Nieznana funkcja: {fn_name}"}

        result_preview = json.dumps(result, ensure_ascii=False)[:300]
        print(f"    ← {result_preview}")

        function_response_parts.append(
            types.Part.from_function_response(name=fn_name, response=result)
        )
        step_calls.append({"name": fn_name, "args": fn_args})

    append_usage_log(
        log_path, usage,
        action="agent_step",
        model=MODEL,
        usage_metrics=usage_metrics,
        cost_usd=cost,
        payload={"step": step + 1, "calls": step_calls},
    )

    history.append(types.Content(role="user", parts=function_response_parts))

    if flag:
        break

# ==============================================================================
# Faza 6: Zapis wyników i podsumowanie
# ==============================================================================

print(f"\n{'=' * 60}")
print(f"Zużycie LLM: {usage['input_tokens']} in / {usage['output_tokens']} out  |  ${usage['cost_usd']:.4f}")

if flag:
    result_path = DATA_DIR / "result.txt"
    result_path.write_text(flag, encoding="utf-8")
    print(f"Flaga: {flag}")
    print(f"Zapisano → {result_path}")
else:
    print(f"Brak flagi po {MAX_STEPS} krokach. Sprawdź {log_path}")

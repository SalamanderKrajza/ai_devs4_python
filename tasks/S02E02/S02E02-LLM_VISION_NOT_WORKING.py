import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from tasks.commons.llm_usage import (
    append_usage_log,
    calculate_usage_cost_usd,
    create_run_logs_dir,
    create_usage_summary,
    extract_gemini_usage_metrics,
)


# ==============================================================================
# S02E02 - Electricity - Agentic board analysis and rotation
# Faza 0: Configuration and environment
# ==============================================================================

load_dotenv()

AI_DEVS_API_KEY = os.environ["AI_DEVS_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

BASE_URL = "https://hub.ag3nts.org"
TASK = "electricity"
VERIFY_URL = f"{BASE_URL}/verify"
CURRENT_BOARD_URL = f"{BASE_URL}/data/{AI_DEVS_API_KEY}/electricity.png"
TARGET_BOARD_URL = f"{BASE_URL}/i/solved_electricity.png"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_BOARD_PATH = DATA_DIR / "current_board.png"
TARGET_BOARD_PATH = DATA_DIR / "target_board.png"
CURRENT_BOARD_MAP_PATH = DATA_DIR / "current_board_map.json"
TARGET_BOARD_MAP_PATH = DATA_DIR / "target_board_map.json"
VERIFY_BOARD_MAP_PATH = DATA_DIR / "verified_board_map.json"
RESULT_PATH = DATA_DIR / "result.txt"
SUMMARY_PATH = DATA_DIR / "summary.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
REQUEST_SLEEP_SECONDS = 0.3

client = genai.Client(api_key=GEMINI_API_KEY)

COORDINATES = [
    "1-1", "1-2", "1-3",
    "2-1", "2-2", "2-3",
    "3-1", "3-2", "3-3",
]

CELL_SCHEMA = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 4,
    "maxItems": 4,
}

BOARD_SCHEMA = {
    "type": "object",
    "properties": {
        "board_map": {
            "type": "object",
            "properties": {coordinate: CELL_SCHEMA for coordinate in COORDINATES},
            "required": COORDINATES,
        },
        "summary": {"type": "string"},
    },
    "required": ["board_map", "summary"],
}

RUN_LOGS_DIR, RUN_ID = create_run_logs_dir(DATA_DIR, "s02e02")
AGENT_LOG_PATH = RUN_LOGS_DIR / "agent_log.jsonl"
ROTATION_LOG_PATH = RUN_LOGS_DIR / "rotation_log.jsonl"
run_cost_summary = create_usage_summary(GEMINI_MODEL)


# ------------------------------------------------------------------------------
# Faza 1: Board tools
# ------------------------------------------------------------------------------

def extract_flag(payload: dict | str) -> str | None:
    """Extract the task flag from a response payload if present."""
    payload_text = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    match = re.search(r"\{FLG:[^}]+\}", payload_text)
    return match.group(0) if match else None


def get_board(board_kind: str, reset: bool = False) -> dict:
    """Download the current or target board image and save it locally."""
    if board_kind not in {"current", "target"}:
        raise ValueError(f"Unsupported board kind: {board_kind}")

    if board_kind == "current":
        url = CURRENT_BOARD_URL if not reset else f"{CURRENT_BOARD_URL}?reset=1"
        destination = CURRENT_BOARD_PATH
    else:
        url = TARGET_BOARD_URL
        destination = TARGET_BOARD_PATH

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)

    print(
        f"\n[GET_BOARD] Downloaded {board_kind} board"
        f"{' with reset' if reset else ''} -> {destination}"
    )

    result = {
        "board_kind": board_kind,
        "reset": reset,
        "path": str(destination),
        "size_bytes": len(response.content),
        "url": url,
    }
    append_usage_log(AGENT_LOG_PATH, run_cost_summary, "get_board", result, model=None)
    return result


def normalize_board_map(board_map: dict) -> dict[str, list[int]]:
    """Normalize model output to an ordered board map with integer vectors."""
    normalized = {}
    for coordinate in COORDINATES:
        values = board_map[coordinate]
        normalized[coordinate] = [int(value) for value in values]
    return normalized


def format_connections(connections: list[int]) -> str:
    """Render [top, right, bottom, left] as both vector and active direction names."""
    direction_names = ["top", "right", "bottom", "left"]
    active_directions = [
        direction_name
        for direction_name, is_active in zip(direction_names, connections)
        if is_active
    ]
    active_label = ",".join(active_directions) if active_directions else "none"
    return f"{connections} -> {active_label}"


def render_board_map_text(board_map: dict[str, list[int]]) -> str:
    """Create a readable multiline text representation of the board map."""
    rendered_lines = []
    for row_index in range(1, 4):
        for column_index in range(1, 4):
            coordinate = f"{row_index}-{column_index}"
            rendered_lines.append(f"{coordinate}: {format_connections(board_map[coordinate])}")
    return "\n".join(rendered_lines)


def validate_board_map(board_map: dict[str, list[int]]) -> list[str]:
    """Return soft warnings for cells that look suspicious in the analyzed board."""
    warnings = []
    for coordinate, connections in board_map.items():
        exit_count = sum(connections)
        if exit_count == 0:
            warnings.append(f"{coordinate} has no exits: {connections}")
        elif exit_count == 1:
            warnings.append(f"{coordinate} has only one exit: {connections}")
    return warnings


def print_board_analysis(board_kind: str, payload: dict) -> None:
    """Print a readable summary of analyzed board state for interactive execution."""
    board_map = payload["board_map"]
    warnings = validate_board_map(board_map)

    print(f"\n[{board_kind.upper()} BOARD] Gemini model: {GEMINI_MODEL}")
    print(f"[{board_kind.upper()} BOARD] Summary: {payload['summary']}")
    print(f"[{board_kind.upper()} BOARD] Text board map:")
    print(render_board_map_text(board_map))

    if warnings:
        print(f"[{board_kind.upper()} BOARD] Suspicious cells:")
        for warning in warnings:
            print(f"  - {warning}")


def analyze_board(board_kind: str) -> dict:
    """Use Gemini Vision to convert a board image into a directional board map."""
    image_path = CURRENT_BOARD_PATH if board_kind == "current" else TARGET_BOARD_PATH
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for {board_kind}: {image_path}")

    prompt_text = (
        "Analyze this 3x3 electricity puzzle board image.\n"
        "Return JSON with `board_map` and `summary`.\n"
        "Coordinates are row-column from top-left to bottom-right: "
        "1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3.\n"
        "Each board_map value must be [top, right, bottom, left].\n"
        "Use 1 when the cable exits the tile on that side, otherwise 0.\n"
        "Inspect only the tile graphics. Do not infer connectivity from neighbors.\n"
        "These tiles are pipe-like connectors, so prefer visually obvious exits only.\n"
        "Avoid guessing single-exit tiles unless the image clearly shows a dead-end piece.\n"
        "Every list must have exactly four integers."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt_text,
            types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=BOARD_SCHEMA,
        ),
    )

    payload = json.loads(response.text)
    payload["board_map"] = normalize_board_map(payload["board_map"])
    usage_metrics = extract_gemini_usage_metrics(response)
    cost_usd = calculate_usage_cost_usd(GEMINI_MODEL, usage_metrics)

    output_path = CURRENT_BOARD_MAP_PATH if board_kind == "current" else TARGET_BOARD_MAP_PATH
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    append_usage_log(
        AGENT_LOG_PATH,
        run_cost_summary,
        "analyze_board",
        {"board_kind": board_kind, "result": payload},
        model=GEMINI_MODEL,
        usage_metrics=usage_metrics,
        cost_usd=cost_usd,
    )
    print_board_analysis(board_kind=board_kind, payload=payload)

    return payload


def rotate_connections_clockwise(connections: list[int]) -> list[int]:
    """Rotate [top, right, bottom, left] one step clockwise."""
    top, right, bottom, left = connections
    return [left, top, right, bottom]


def rotation_distance(current_connections: list[int], target_connections: list[int]) -> int | None:
    """Return the minimal number of clockwise rotations needed to match the target."""
    rotated = list(current_connections)
    for steps in range(4):
        if rotated == target_connections:
            return steps
        rotated = rotate_connections_clockwise(rotated)
    return None


def build_mismatch_summary(
    current_board_map: dict[str, list[int]],
    target_board_map: dict[str, list[int]],
) -> list[dict]:
    """Compare two board maps and describe the remaining mismatches."""
    mismatches = []
    for coordinate in COORDINATES:
        current_connections = current_board_map[coordinate]
        target_connections = target_board_map[coordinate]
        needed_rotations = rotation_distance(current_connections, target_connections)
        if needed_rotations == 0:
            continue
        mismatch = {
            "cell": coordinate,
            "current": current_connections,
            "target": target_connections,
            "rotations_needed": needed_rotations,
        }
        if needed_rotations is None:
            mismatch["error"] = (
                f"Current tile {current_connections} cannot be rotated into target {target_connections}"
            )
        mismatches.append(mismatch)
    return mismatches


def print_mismatch_summary(mismatches: list[dict]) -> None:
    """Print a readable comparison of the current board against the target board."""
    if not mismatches:
        print("\n[MISMATCH CHECK] Current board matches the target board.")
        return

    print("\n[MISMATCH CHECK] Remaining differences:")
    for mismatch in mismatches:
        if mismatch["rotations_needed"] is None:
            print(
                "  - "
                f"{mismatch['cell']}: current={mismatch['current']} target={mismatch['target']} "
                f"-> IMPOSSIBLE ROTATION ({mismatch['error']})"
            )
        else:
            print(
                "  - "
                f"{mismatch['cell']}: current={mismatch['current']} target={mismatch['target']} "
                f"-> rotate x{mismatch['rotations_needed']}"
            )


def calculate_required_rotations(
    current_board_map: dict[str, list[int]],
    target_board_map: dict[str, list[int]],
) -> dict:
    """Calculate the full clockwise rotation plan from current board to target board."""
    mismatches = build_mismatch_summary(current_board_map, target_board_map)
    print_mismatch_summary(mismatches)

    impossible_mismatches = [
        mismatch for mismatch in mismatches if mismatch["rotations_needed"] is None
    ]
    if impossible_mismatches:
        return {
            "solvable": False,
            "error": "Unable to get desired pattern. Probably image_analyze went wrong.",
            "mismatches": mismatches,
            "rotation_plan": [],
        }

    rotation_plan = []
    for mismatch in mismatches:
        rotation_plan.extend([mismatch["cell"]] * mismatch["rotations_needed"])

    result = {
        "solvable": True,
        "mismatches": mismatches,
        "rotation_plan": rotation_plan,
    }
    append_usage_log(
        AGENT_LOG_PATH,
        run_cost_summary,
        "calculate_required_rotations",
        result,
        model=None,
    )
    return result


def to_api_coordinate(cell: str) -> str:
    """Convert internal `row-col` cell notation into API `rowxcol` notation."""
    row, column = cell.split("-")
    return f"{row}x{column}"


def rotate(cell: str) -> dict:
    """Rotate one tile clockwise by calling the task API once."""
    api_cell = to_api_coordinate(cell)
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"rotate": api_cell},
    }
    response = requests.post(VERIFY_URL, json=payload, timeout=30)
    response.raise_for_status()
    response_payload = response.json()

    log_entry = {
        "cell": cell,
        "api_cell": api_cell,
        "response": response_payload,
        "flag": extract_flag(response_payload),
        "timestamp": time.time(),
    }
    with open(ROTATION_LOG_PATH, "a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    append_usage_log(AGENT_LOG_PATH, run_cost_summary, "rotate", log_entry, model=None)

    time.sleep(REQUEST_SLEEP_SECONDS)
    return log_entry


def verify(board_state: dict) -> dict:
    """
    Refresh the current board, analyze it again, and compare it with the target board.
    """
    get_board_result = get_board("current", reset=False)
    current_analysis = analyze_board("current")
    board_state["current_board_map"] = current_analysis["board_map"]
    board_state["board_summaries"]["current"] = current_analysis["summary"]

    mismatches = build_mismatch_summary(
        board_state["current_board_map"],
        board_state["target_board_map"],
    )
    verification_result = {
        "get_board": get_board_result,
        "current_board_map": current_analysis["board_map"],
        "summary": current_analysis["summary"],
        "mismatches": mismatches,
        "is_match": not mismatches,
        "flag": board_state.get("flag"),
    }
    append_usage_log(AGENT_LOG_PATH, run_cost_summary, "verify", verification_result, model=None)
    print_mismatch_summary(mismatches)

    VERIFY_BOARD_MAP_PATH.write_text(
        json.dumps(verification_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return verification_result


# ------------------------------------------------------------------------------
# Faza 2: Single-pass analysis + deterministic rotation plan
# ------------------------------------------------------------------------------

board_state = {
    "current_board_map": {},
    "target_board_map": {},
    "board_summaries": {},
    "downloaded_boards": {},
    "rotation_history": [],
    "flag": None,
}

print("\n[PIPELINE] Starting low-cost board workflow.")

board_state["downloaded_boards"]["current"] = get_board("current")
current_analysis = analyze_board("current")
board_state["current_board_map"] = current_analysis["board_map"]
board_state["board_summaries"]["current"] = current_analysis["summary"]

board_state["downloaded_boards"]["target"] = get_board("target")
target_analysis = analyze_board("target")
board_state["target_board_map"] = target_analysis["board_map"]
board_state["board_summaries"]["target"] = target_analysis["summary"]

rotation_plan_result = calculate_required_rotations(
    board_state["current_board_map"],
    board_state["target_board_map"],
)

if not rotation_plan_result["solvable"]:
    raise ValueError(
        rotation_plan_result["error"] + "\n"
        + json.dumps(rotation_plan_result["mismatches"], ensure_ascii=False, indent=2)
    )

print(f"\n[ROTATION PLAN] Planned moves: {rotation_plan_result['rotation_plan']}")

for step_index, cell in enumerate(
    tqdm(rotation_plan_result["rotation_plan"], desc="Applying rotations"),
    start=1,
):
    print(f"[ROTATE] Step {step_index}: rotating cell {cell}")
    rotate_result = rotate(cell)
    board_state["rotation_history"].append(cell)
    board_state["current_board_map"][cell] = rotate_connections_clockwise(
        board_state["current_board_map"][cell]
    )
    if rotate_result["flag"]:
        board_state["flag"] = rotate_result["flag"]
        break


# ------------------------------------------------------------------------------
# Faza 3: One final verification
# ------------------------------------------------------------------------------

verification_result = verify(board_state)


# ------------------------------------------------------------------------------
# Faza 4: Persist final result
# ------------------------------------------------------------------------------

final_summary = {
    "run_id": RUN_ID,
    "logs_dir": str(RUN_LOGS_DIR),
    "flag": board_state["flag"],
    "rotation_history": board_state["rotation_history"],
    "current_board_map": board_state["current_board_map"],
    "target_board_map": board_state["target_board_map"],
    "verification_result": verification_result,
    "rotation_plan": rotation_plan_result,
    "usage_summary": run_cost_summary,
}

SUMMARY_PATH.write_text(
    json.dumps(final_summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

if board_state["flag"]:
    RESULT_PATH.write_text(board_state["flag"], encoding="utf-8")
    print(f"FLAG: {board_state['flag']}")
else:
    RESULT_PATH.write_text("No flag captured.", encoding="utf-8")
    print("No flag captured.")

print("\nUsage summary:")
print(json.dumps(run_cost_summary, ensure_ascii=False, indent=2))
print("\nFinal verification:")
print(json.dumps(verification_result, ensure_ascii=False, indent=2))

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ==============================================================================
# S02E02v2 - Electricity - Fully programmatic CV pipeline
# ==============================================================================

load_dotenv()

AI_DEVS_API_KEY = os.environ["AI_DEVS_API_KEY"]

BASE_URL = "https://hub.ag3nts.org"
TASK = "electricity"
VERIFY_URL = f"{BASE_URL}/verify"
CURRENT_BOARD_URL = f"{BASE_URL}/data/{AI_DEVS_API_KEY}/electricity.png"
TARGET_BOARD_URL = f"{BASE_URL}/i/solved_electricity.png"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = f"s02e02v2_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
RUN_DIR = DATA_DIR / "logs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_BOARD_PATH = RUN_DIR / "current_board.png"
TARGET_BOARD_PATH = RUN_DIR / "target_board.png"
CURRENT_WARPED_PATH = RUN_DIR / "current_board_warped.png"
TARGET_WARPED_PATH = RUN_DIR / "target_board_warped.png"
CURRENT_MASK_PATH = RUN_DIR / "current_board_mask.png"
TARGET_MASK_PATH = RUN_DIR / "target_board_mask.png"
CURRENT_MAP_PATH = RUN_DIR / "current_board_map.json"
TARGET_MAP_PATH = RUN_DIR / "target_board_map.json"
ROTATION_PLAN_PATH = RUN_DIR / "rotation_plan.json"
VERIFY_RESULT_PATH = RUN_DIR / "verify_result.json"
SUMMARY_PATH = RUN_DIR / "summary.json"
RESULT_PATH = RUN_DIR / "result.txt"

BOARD_SIZE = 900
GRID_SIZE = 3
TILE_SIZE = BOARD_SIZE // GRID_SIZE
TILE_INNER_MARGIN = 18
EDGE_PROBE_WIDTH_RATIO = 0.24
EDGE_PROBE_DEPTH_RATIO = 0.16
EDGE_ACTIVATION_THRESHOLD = 0.18
ROTATION_SIMILARITY_THRESHOLD = 0.55
REQUEST_SLEEP_SECONDS = 0.25

DIRECTION_ORDER = ["top", "right", "bottom", "left"]
COORDINATES = [
    "1-1", "1-2", "1-3",
    "2-1", "2-2", "2-3",
    "3-1", "3-2", "3-3",
]


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def to_api_coordinate(cell: str) -> str:
    """Convert internal row-col notation into API rowxcol notation."""
    row, column = cell.split("-")
    return f"{row}x{column}"


def rotate_connections_clockwise(connections: list[int]) -> list[int]:
    """Rotate [top, right, bottom, left] one step clockwise."""
    top, right, bottom, left = connections
    return [left, top, right, bottom]


def rotation_distance(current_connections: list[int], target_connections: list[int]) -> int | None:
    """Return the number of clockwise rotations needed to match target or None."""
    rotated = list(current_connections)
    for steps in range(4):
        if rotated == target_connections:
            return steps
        rotated = rotate_connections_clockwise(rotated)
    return None


def format_connections(connections: list[int]) -> str:
    """Render [top, right, bottom, left] as active directions."""
    active_directions = [
        direction_name
        for direction_name, is_active in zip(DIRECTION_ORDER, connections)
        if is_active
    ]
    return f"{connections} -> {','.join(active_directions) if active_directions else 'none'}"


def print_board_map(board_name: str, board_map: dict[str, list[int]]) -> None:
    """Print a readable board map for interactive inspection."""
    print(f"\n[{board_name}] Board map")
    for coordinate in COORDINATES:
        print(f"  {coordinate}: {format_connections(board_map[coordinate])}")


def save_json(target_path: Path, payload: dict) -> None:
    """Save JSON payload in UTF-8."""
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------------
# Fetch task images
# ------------------------------------------------------------------------------

def download_image(url: str, target_path: Path) -> Path:
    """Download one image file from the hub."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    print(f"[DOWNLOAD] Saved {url} -> {target_path}")
    return target_path


# ------------------------------------------------------------------------------
# Board detection and perspective normalization
# ------------------------------------------------------------------------------

def load_image(image_path: Path) -> np.ndarray:
    """Load image with OpenCV and fail fast if the file is unreadable."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def build_board_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Create a high-contrast binary mask of the dark board structure."""
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, binary_inverse = cv2.threshold(
        image_blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((5, 5), dtype=np.uint8)
    binary_closed = cv2.morphologyEx(binary_inverse, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary_opened


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return 4 quad points ordered as top-left, top-right, bottom-right, bottom-left."""
    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def contour_to_quad(contour: np.ndarray) -> np.ndarray:
    """Approximate a contour with 4 points, or derive 4 corners from minAreaRect."""
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approximation) == 4:
        return approximation.reshape(4, 2)

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect)


def find_board_quad(mask: np.ndarray) -> np.ndarray:
    """Find the most likely board contour and return it as a 4-point quadrilateral."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found while trying to detect the board.")

    image_area = mask.shape[0] * mask.shape[1]
    candidate_quads = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue

        quad = contour_to_quad(contour)
        candidate_quads.append((area, quad))

    if not candidate_quads:
        raise ValueError("No sufficiently large contour found for the board.")

    _, best_quad = max(candidate_quads, key=lambda item: item[0])
    return order_quad_points(best_quad)


def warp_board(image_bgr: np.ndarray, quad: np.ndarray, size: int = BOARD_SIZE) -> np.ndarray:
    """Warp detected board into a square top-down view."""
    destination = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(image_bgr, transform, (size, size))


def detect_and_warp_board(source_path: Path, warped_path: Path, mask_path: Path) -> np.ndarray:
    """Detect board bounds automatically and save a normalized warped board image."""
    image_bgr = load_image(source_path)
    board_mask = build_board_mask(image_bgr)
    board_quad = find_board_quad(board_mask)
    board_warped = warp_board(image_bgr, board_quad)

    cv2.imwrite(str(mask_path), board_mask)
    cv2.imwrite(str(warped_path), board_warped)
    print(f"[WARP] Saved normalized board -> {warped_path}")
    return board_warped


# ------------------------------------------------------------------------------
# Tile cropping and tile-to-map conversion
# ------------------------------------------------------------------------------

def crop_tiles(board_warped: np.ndarray, board_name: str) -> dict[str, np.ndarray]:
    """Split a normalized board image into 9 equal tiles and save them for debugging."""
    tiles = {}
    for row_index in range(GRID_SIZE):
        for column_index in range(GRID_SIZE):
            coordinate = f"{row_index + 1}-{column_index + 1}"
            y0 = row_index * TILE_SIZE
            y1 = (row_index + 1) * TILE_SIZE
            x0 = column_index * TILE_SIZE
            x1 = (column_index + 1) * TILE_SIZE
            tile = board_warped[y0:y1, x0:x1].copy()
            tiles[coordinate] = tile

            tile_path = RUN_DIR / f"{board_name}_tile_{coordinate}.png"
            cv2.imwrite(str(tile_path), tile)

    return tiles


def tile_to_binary(tile_bgr: np.ndarray) -> np.ndarray:
    """Convert one tile into a binary mask emphasizing dark connector strokes."""
    tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    tile_blurred = cv2.GaussianBlur(tile_gray, (5, 5), 0)
    _, tile_binary_inverse = cv2.threshold(
        tile_blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)
    tile_binary_inverse = cv2.morphologyEx(
        tile_binary_inverse,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )
    return tile_binary_inverse


def build_edge_probe_slices(tile_shape: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    """Create probe windows near the center of each edge."""
    height, width = tile_shape
    probe_half_width = max(8, int(width * EDGE_PROBE_WIDTH_RATIO / 2))
    probe_depth = max(10, int(height * EDGE_PROBE_DEPTH_RATIO))
    center_x = width // 2
    center_y = height // 2

    return {
        "top": (
            slice(TILE_INNER_MARGIN, TILE_INNER_MARGIN + probe_depth),
            slice(center_x - probe_half_width, center_x + probe_half_width),
        ),
        "right": (
            slice(center_y - probe_half_width, center_y + probe_half_width),
            slice(width - TILE_INNER_MARGIN - probe_depth, width - TILE_INNER_MARGIN),
        ),
        "bottom": (
            slice(height - TILE_INNER_MARGIN - probe_depth, height - TILE_INNER_MARGIN),
            slice(center_x - probe_half_width, center_x + probe_half_width),
        ),
        "left": (
            slice(center_y - probe_half_width, center_y + probe_half_width),
            slice(TILE_INNER_MARGIN, TILE_INNER_MARGIN + probe_depth),
        ),
    }


def detect_tile_exits(tile_bgr: np.ndarray, coordinate: str, board_name: str) -> list[int]:
    """Detect cable exits on one tile by sampling binary mask windows near each edge."""
    tile_binary = tile_to_binary(tile_bgr)
    probe_slices = build_edge_probe_slices(tile_binary.shape)

    debug_tile_path = RUN_DIR / f"{board_name}_tile_mask_{coordinate}.png"
    cv2.imwrite(str(debug_tile_path), tile_binary)

    exits = []
    for direction in DIRECTION_ORDER:
        row_slice, col_slice = probe_slices[direction]
        probe = tile_binary[row_slice, col_slice]
        activation_ratio = float(np.count_nonzero(probe)) / float(probe.size)
        exits.append(1 if activation_ratio >= EDGE_ACTIVATION_THRESHOLD else 0)

    return exits


def build_board_map(board_warped: np.ndarray, board_name: str) -> dict[str, list[int]]:
    """Build a directional board map for the whole 3x3 board."""
    tiles = crop_tiles(board_warped, board_name=board_name)
    board_map = {}

    for coordinate, tile in tiles.items():
        board_map[coordinate] = detect_tile_exits(tile, coordinate=coordinate, board_name=board_name)

    print_board_map(board_name, board_map)
    return board_map


# ------------------------------------------------------------------------------
# Rotation planning
# ------------------------------------------------------------------------------

def calculate_required_rotations(
    current_board_map: dict[str, list[int]],
    target_board_map: dict[str, list[int]],
) -> dict:
    """Calculate a deterministic clockwise rotation plan."""
    mismatches = []
    rotation_plan = []

    for coordinate in COORDINATES:
        current_connections = current_board_map[coordinate]
        target_connections = target_board_map[coordinate]
        needed_rotations = rotation_distance(current_connections, target_connections)

        mismatch = {
            "cell": coordinate,
            "current": current_connections,
            "target": target_connections,
            "rotations_needed": needed_rotations,
        }
        if needed_rotations is None:
            mismatch["error"] = (
                f"Unable to rotate {current_connections} into {target_connections} at {coordinate}"
            )
            mismatches.append(mismatch)
            continue

        if needed_rotations > 0:
            mismatches.append(mismatch)
            rotation_plan.extend([coordinate] * needed_rotations)

    return {
        "solvable": not any(item["rotations_needed"] is None for item in mismatches),
        "mismatches": mismatches,
        "rotation_plan": rotation_plan,
    }


# ------------------------------------------------------------------------------
# Hub interaction
# ------------------------------------------------------------------------------

def rotate_tile(cell: str) -> dict:
    """Send one clockwise rotation request to the hub."""
    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": TASK,
        "answer": {"rotate": to_api_coordinate(cell)},
    }
    response = requests.post(VERIFY_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_flag(payload: dict | str) -> str | None:
    """Extract the task flag if present in payload."""
    payload_text = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    start = payload_text.find("{FLG:")
    if start == -1:
        return None
    end = payload_text.find("}", start)
    if end == -1:
        return None
    return payload_text[start : end + 1]


# ------------------------------------------------------------------------------
# Execute pipeline
# ------------------------------------------------------------------------------

download_image(CURRENT_BOARD_URL, CURRENT_BOARD_PATH)
download_image(TARGET_BOARD_URL, TARGET_BOARD_PATH)

current_board_warped = detect_and_warp_board(
    CURRENT_BOARD_PATH,
    warped_path=CURRENT_WARPED_PATH,
    mask_path=CURRENT_MASK_PATH,
)
target_board_warped = detect_and_warp_board(
    TARGET_BOARD_PATH,
    warped_path=TARGET_WARPED_PATH,
    mask_path=TARGET_MASK_PATH,
)

current_board_map = build_board_map(current_board_warped, board_name="current")
target_board_map = build_board_map(target_board_warped, board_name="target")

save_json(CURRENT_MAP_PATH, {"board_map": current_board_map})
save_json(TARGET_MAP_PATH, {"board_map": target_board_map})

rotation_plan_result = calculate_required_rotations(
    current_board_map=current_board_map,
    target_board_map=target_board_map,
)
save_json(ROTATION_PLAN_PATH, rotation_plan_result)

print("\n[PLAN] Rotation plan:")
print(json.dumps(rotation_plan_result, ensure_ascii=False, indent=2))

if not rotation_plan_result["solvable"]:
    raise ValueError(
        "Automatic board mapping is inconsistent; at least one tile cannot be rotated into the target.\n"
        + json.dumps(rotation_plan_result["mismatches"], ensure_ascii=False, indent=2)
    )

flag = None
rotation_responses = []
for step_index, cell in enumerate(
    tqdm(rotation_plan_result["rotation_plan"], desc="Applying rotations"),
    start=1,
):
    print(f"[ROTATE] Step {step_index}: {cell} -> {to_api_coordinate(cell)}")
    rotate_response = rotate_tile(cell)
    rotation_responses.append(
        {
            "step": step_index,
            "cell": cell,
            "api_cell": to_api_coordinate(cell),
            "response": rotate_response,
        }
    )
    flag = extract_flag(rotate_response) or flag
    if flag:
        break

    current_board_map[cell] = rotate_connections_clockwise(current_board_map[cell])
    save_json(RUN_DIR / "rotation_responses.json", {"responses": rotation_responses})

    if REQUEST_SLEEP_SECONDS:
        import time

        time.sleep(REQUEST_SLEEP_SECONDS)

verify_payload = {
    "run_id": RUN_ID,
    "flag": flag,
    "rotation_plan": rotation_plan_result,
    "rotation_responses": rotation_responses,
}
save_json(VERIFY_RESULT_PATH, verify_payload)

summary = {
    "run_id": RUN_ID,
    "run_dir": str(RUN_DIR),
    "current_board_map": current_board_map,
    "target_board_map": target_board_map,
    "rotation_plan_result": rotation_plan_result,
    "flag": flag,
}
save_json(SUMMARY_PATH, summary)

if flag:
    RESULT_PATH.write_text(flag, encoding="utf-8")
    print(f"\nFLAG: {flag}")
else:
    RESULT_PATH.write_text("No flag captured.", encoding="utf-8")
    print("\nNo flag captured.")

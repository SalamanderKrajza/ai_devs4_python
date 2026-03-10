import json
import math
import os
import sys
import time
from pathlib import Path

import requests
from google import genai
from google.genai import types
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tasks.commons.task_handler import ai_devs_key, send_verify


# --------------------------------------------------------------
# Validate environment
# --------------------------------------------------------------
gemini_api_key = os.environ["GEMINI_API_KEY"]


# --------------------------------------------------------------
# Load suspects from S01E01 output
# --------------------------------------------------------------
base_dir = Path(__file__).resolve().parent
suspects_json_path = base_dir.parent / "S01E01" / "data" / "suspects.json"
suspects = json.loads(suspects_json_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------
# Fetch power plant locations
# --------------------------------------------------------------
power_plants_url = f"https://hub.ag3nts.org/data/{ai_devs_key}/findhim_locations.json"
power_plants_response = requests.get(power_plants_url, timeout=30)
power_plants_response.raise_for_status()
power_plants_raw = power_plants_response.json()

data_dir = base_dir / "data"
def geocode_location(name: str) -> tuple[float, float]:
    query = f"{name}, Poland"
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": "ai-devs-s02e02"},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"No geocoding results for {name}")
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    time.sleep(1)
    return lat, lon

power_plants_extended = []
for plant_name, plant_data in tqdm(
    power_plants_raw["power_plants"].items(),
    total=len(power_plants_raw["power_plants"]),
    desc="Geocoding power plants",
):
    lat = plant_data.get("lat")
    lon = plant_data.get("lon")
    if lat is None or lon is None:
        lat, lon = geocode_location(plant_name)
    power_plants_extended.append(
        {
            "name": plant_name,
            "code": plant_data["code"],
            "power": plant_data.get("power"),
            "is_active": plant_data.get("is_active"),
            "lat": lat,
            "lon": lon,
        }
    )

data_dir.mkdir(parents=True, exist_ok=True)
power_plants_path = data_dir / "power_plants.json"
power_plants_path.write_text(
    json.dumps(power_plants_raw, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

power_plants_extended_path = data_dir / "power_plants_extended.json"
power_plants_extended_path.write_text(
    json.dumps(power_plants_extended, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

power_plants = power_plants_extended

# --------------------------------------------------------------
# Distance utility (Haversine)
# --------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def extract_coordinates(item: dict) -> tuple[float, float]:
    lat_key = next((key for key in ["lat", "latitude"] if key in item), None)
    lon_key = next((key for key in ["lon", "lng", "longitude"] if key in item), None)
    if not lat_key or not lon_key:
        raise KeyError(f"Missing lat/lon in location item: {item}")
    return float(item[lat_key]), float(item[lon_key])


def normalize_locations(payload: object) -> list[dict]:
    if isinstance(payload, dict) and "locations" in payload:
        return payload["locations"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unexpected locations payload: {payload}")


# --------------------------------------------------------------
# API helpers
# --------------------------------------------------------------
location_url = "https://hub.ag3nts.org/api/location"
access_url = "https://hub.ag3nts.org/api/accesslevel"


def get_locations(name: str, surname: str) -> list[dict]:
    payload = {"apikey": ai_devs_key, "name": name, "surname": surname}
    response = requests.post(location_url, json=payload, timeout=30)
    response.raise_for_status()
    return normalize_locations(response.json())


def get_access_level(name: str, surname: str, birth_year: int) -> int:
    payload = {
        "apikey": ai_devs_key,
        "name": name,
        "surname": surname,
        "birthYear": int(birth_year),
    }
    response = requests.post(access_url, json=payload, timeout=30)
    response.raise_for_status()
    response_json = response.json()
    if isinstance(response_json, dict) and "accessLevel" in response_json:
        return int(response_json["accessLevel"])
    return int(response_json)


def find_closest_power_plant(locations: list[dict]) -> tuple[str, float]:
    best_plant_code = None
    best_distance = None

    for plant in power_plants:
        plant_lat, plant_lon = extract_coordinates(plant)
        plant_code = plant.get("code") or plant.get("id") or plant.get("name")

        for location in locations:
            loc_lat, loc_lon = extract_coordinates(location)
            distance = haversine_km(loc_lat, loc_lon, plant_lat, plant_lon)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_plant_code = plant_code

    if best_plant_code is None or best_distance is None:
        raise ValueError("No valid plant distance computed.")
    return best_plant_code, best_distance


# --------------------------------------------------------------
# Agent loop (Gemini Flash 2.5)
# --------------------------------------------------------------
gemini_model = "gemini-2.5-flash"
client = genai.Client(api_key=gemini_api_key)

agent_system = (
    "You are an agent that must find the suspect closest to a power plant. "
    "At each step choose one action: "
    "get_locations, get_access_level, finish. "
    "Process suspects one by one, update the best candidate using the state. "
    "Only ask for access level when the best candidate is known. "
    "Finish only when access level is known."
)

agent_schema = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["get_locations", "get_access_level", "finish"],
        },
        "suspect_index": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

processed_indices: set[int] = set()
best_candidate = None
best_distance_km = None
best_power_plant = None
best_candidate_access_level = None
max_steps = 12

for step_idx in tqdm(range(max_steps), desc="Agent steps", total=max_steps):
    state_summary = {
        "total_suspects": len(suspects),
        "processed_indices": sorted(processed_indices),
        "best_candidate": best_candidate,
        "best_distance_km": best_distance_km,
        "best_power_plant": best_power_plant,
        "access_level_known": best_candidate_access_level is not None,
    }
    user_text = (
        "State:\n"
        f"{json.dumps(state_summary, ensure_ascii=False)}\n\n"
        "Suspects:\n"
        f"{json.dumps(suspects, ensure_ascii=False)}"
    )

    response = client.models.generate_content(
        model=gemini_model,
        contents=user_text,
        config=types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=agent_system,
            response_mime_type="application/json",
            response_schema=agent_schema,
        ),
    )

    action_payload = json.loads(response.text)
    action = action_payload["action"]
    suspect_index = action_payload.get("suspect_index")

    if action == "get_locations":
        if suspect_index is None or suspect_index in processed_indices:
            continue

        suspect = suspects[suspect_index]
        locations = get_locations(suspect["name"], suspect["surname"])
        locations_path = data_dir / f"locations_{suspect['surname']}_{suspect['name']}.json"
        locations_path.write_text(
            json.dumps(locations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        plant_code, distance_km = find_closest_power_plant(locations)
        processed_indices.add(suspect_index)

        if best_distance_km is None or distance_km < best_distance_km:
            best_distance_km = distance_km
            best_power_plant = plant_code
            best_candidate = suspect

    elif action == "get_access_level":
        if best_candidate is None or best_candidate_access_level is not None:
            continue

        best_candidate_access_level = get_access_level(
            best_candidate["name"],
            best_candidate["surname"],
            best_candidate["birthYear"],
        )

    elif action == "finish":
        if best_candidate and best_candidate_access_level is not None:
            break


# --------------------------------------------------------------
# Prepare and send verify payload
# --------------------------------------------------------------
result_payload = {
    "name": best_candidate["name"],
    "surname": best_candidate["surname"],
    "accessLevel": best_candidate_access_level,
    "powerPlant": best_power_plant,
    "distanceKm": best_distance_km,
}

result_path = data_dir / "result.json"
result_path.write_text(
    json.dumps(result_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

verify_answer = {
    "name": best_candidate["name"],
    "surname": best_candidate["surname"],
    "accessLevel": best_candidate_access_level,
    "powerPlant": best_power_plant,
}
verify_payload = {"apikey": ai_devs_key, "task": "findhim", "answer": verify_answer}
verify_response = send_verify(verify_payload)

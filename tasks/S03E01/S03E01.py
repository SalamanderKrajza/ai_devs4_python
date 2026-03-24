# ==============================================================================
# S03E01 - evaluation - Anomalie w odczytach sensorów elektrowni
# Faza 0: Konfiguracja i importy
# ==============================================================================

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

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

SENSORS_ZIP_URL = "https://hub.ag3nts.org/dane/sensors.zip"
TASK = "evaluation"
LLM_MODEL = "gpt-4o-mini"
BATCH_SIZE = 100

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SENSORS_DIR = DATA_DIR / "sensors"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SENSORS_DIR.mkdir(parents=True, exist_ok=True)

run_dir, run_id = create_run_logs_dir(DATA_DIR, "s03e01")
log_path = run_dir / "llm_log.jsonl"
session_usage = create_usage_summary(LLM_MODEL)

client = OpenAI()

print(f"Run ID: {run_id}")
print(f"Model: {LLM_MODEL}, batch: {BATCH_SIZE}")

VALID_RANGES = {
    "temperature_K":      (553,   873),
    "pressure_bar":       (60,    160),
    "water_level_meters": (5.0,   15.0),
    "voltage_supply_v":   (229.0, 231.0),
    "humidity_percent":   (40.0,  80.0),
}

SENSOR_KEYWORD_TO_FIELD = {
    "temperature": "temperature_K",
    "pressure":    "pressure_bar",
    "water":       "water_level_meters",
    "voltage":     "voltage_supply_v",
    "humidity":    "humidity_percent",
}

NUMERIC_FIELDS = list(VALID_RANGES.keys())
SENSOR_KEYWORDS = list(SENSOR_KEYWORD_TO_FIELD.keys())

# ==============================================================================
# Faza 1: Pobieranie i rozpakowywanie sensors.zip
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 1: Pobieranie sensors.zip...")

zip_cache = DATA_DIR / "sensors.zip"
if not zip_cache.exists():
    resp = requests.get(SENSORS_ZIP_URL, timeout=120, stream=True)
    resp.raise_for_status()
    with open(zip_cache, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    print(f"  Pobrano -> {zip_cache} ({zip_cache.stat().st_size // 1024} KB)")
else:
    print(f"  Cache: {zip_cache}")

json_files = list(SENSORS_DIR.glob("*.json"))
if not json_files:
    print("  Rozpakowywanie...")
    with zipfile.ZipFile(zip_cache, "r") as zf:
        zf.extractall(SENSORS_DIR)
    json_files = list(SENSORS_DIR.glob("*.json"))

print(f"  Pliki JSON: {len(json_files)}")

# ==============================================================================
# Faza 2: Wczytanie + dwa oddzielne indeksy deduplication
#
#   data_hash = hash(sensor_type + wartości numeryczne)
#               → dla anomalii programistycznych (identyczne pomiary = ten sam wynik)
#
#   note_hash = hash(operator_notes)
#               → dla LLM (identyczna notatka = ta sama etykieta)
#               Jedna notatka może pojawić się przy różnych sensor_type,
#               dlatego mismatch sprawdzamy osobno na poziomie pliku.
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 2: Wczytywanie i budowanie indeksow...")


def make_data_hash(data: dict) -> str:
    key = {"sensor_type": data.get("sensor_type", "")}
    for f in NUMERIC_FIELDS:
        key[f] = data.get(f, 0)
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def make_note_hash(data: dict) -> str:
    note = data.get("operator_notes", "").strip()
    return hashlib.sha256(note.encode()).hexdigest()


data_hash_to_ids:  dict[str, list[str]] = {}
data_hash_to_data: dict[str, dict]      = {}

note_hash_to_ids:  dict[str, list[str]] = {}
note_hash_to_note: dict[str, str]       = {}

# Per-file lookups potrzebne w fazie 5 do scoringu
file_data_hash:   dict[str, str] = {}
file_note_hash:   dict[str, str] = {}
file_sensor_type: dict[str, str] = {}

for json_path in tqdm(sorted(json_files), desc="Wczytywanie"):
    file_id = json_path.stem
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARN: {json_path.name}: {e}")
        continue

    dh = make_data_hash(data)
    nh = make_note_hash(data)

    file_data_hash[file_id]   = dh
    file_note_hash[file_id]   = nh
    file_sensor_type[file_id] = data.get("sensor_type", "")

    if dh not in data_hash_to_ids:
        data_hash_to_ids[dh]  = []
        data_hash_to_data[dh] = data
    data_hash_to_ids[dh].append(file_id)

    if nh not in note_hash_to_ids:
        note_hash_to_ids[nh]  = []
        note_hash_to_note[nh] = data.get("operator_notes", "").strip()
    note_hash_to_ids[nh].append(file_id)

n_files = len(file_data_hash)
print(f"  Lacznie plikow:              {n_files}")
print(f"  Unikalne kombinacje danych:  {len(data_hash_to_ids)}  (duplikaty: {n_files - len(data_hash_to_ids)})")
print(f"  Unikalne notatki:            {len(note_hash_to_ids)}  (duplikaty: {n_files - len(note_hash_to_ids)})")

# Unikalne typy sensorów wyciągnięte z danych — używane w prompcie LLM
observed_sensor_keywords = sorted({
    kw.strip()
    for st in file_sensor_type.values()
    for kw in st.lower().split("/")
    if kw.strip()
})
print(f"  Wykryte typy sensorow:       {observed_sensor_keywords}")

# ==============================================================================
# Faza 3: Programistyczna detekcja anomalii danych
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 3: Programistyczna detekcja anomalii danych...")


def get_active_fields(sensor_type: str) -> set[str]:
    return {
        field
        for keyword, field in SENSOR_KEYWORD_TO_FIELD.items()
        if keyword in sensor_type.lower()
    }


def check_data_anomaly(data: dict) -> bool:
    sensor_type = data.get("sensor_type", "")
    active = get_active_fields(sensor_type)
    for field, (lo, hi) in VALID_RANGES.items():
        val = data.get(field, 0)
        if field in active:
            if val < lo or val > hi:
                return True   # aktywny sensor poza zakresem (typ 1)
        else:
            if val != 0:
                return True   # nieaktywny sensor z niezerowa wartoscia (typ 4)
    return False


data_bad: set[str] = set()   # data_hashes z anomalii danych
data_ok:  set[str] = set()

for dh, data in tqdm(data_hash_to_data.items(), desc="Analiza danych"):
    if check_data_anomaly(data):
        data_bad.add(dh)
    else:
        data_ok.add(dh)

print(f"  Anomalie danych (unikalne data_hash):  {len(data_bad)}")
print(f"  OK danych (unikalne data_hash):        {len(data_ok)}")

# ==============================================================================
# Faza 4: LLM – etykietowanie wszystkich unikalnych notatek
#
# Każda notatka otrzymuje 3 flagi:
#   sentiment      – "ok" | "problem" | "neutral"
#   sensor_mentions – lista typów sensorów jawnie wymienionych w notatce
#   coherent       – czy notatka jest sensownym, czytelnym zdaniem
#
# LLM dostaje nazwy 5 typów sensorów — potrzebne tylko do sensor_mentions.
# Zakresy wartości są bez znaczenia — oceniamy je programistycznie.
#
# Wyniki są cache'owane, żeby nie przepalać tokenów przy ponownym uruchomieniu.
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 4: LLM – etykietowanie notatek operatora...")
print(f"  Unikalne notatki do etykietowania: {len(note_hash_to_ids)}")

LABELS_CACHE = DATA_DIR / "note_labels.json"

sensor_keywords_str = ", ".join(observed_sensor_keywords)

SYSTEM_PROMPT = f"""You are analyzing operator notes from a nuclear power plant sensor monitoring system.

The plant uses the following sensor types (may be combined with "/"):
  {sensor_keywords_str}

For each operator note assign three labels:

1. sentiment:
   "ok"      – operator explicitly states readings are normal / stable / within range / no issues
   "problem" – operator reports an error, fault, anomaly, malfunction, unusual reading, or any concern
   "neutral" – no clear statement about system state (e.g. purely procedural log entry)

2. sensor_mentions: list of sensor keywords from [temperature, pressure, water, voltage, humidity]
   that are explicitly referenced in the note. Use [] if the note is generic with no specific sensor reference.

3. coherent: true  – the note is a meaningful, readable human sentence
             false – gibberish, random characters, or logically nonsensical text

Return ONLY a JSON array, one object per input entry, indexed by idx:
[{{"idx": 0, "sentiment": "ok", "sensor_mentions": [], "coherent": true}}, ...]"""

# Wczytaj cache jeśli istnieje
if LABELS_CACHE.exists():
    print(f"  Cache: {LABELS_CACHE}")
    raw_labels: dict[str, dict] = json.loads(LABELS_CACHE.read_text(encoding="utf-8"))
    # Klucze w cache to note_hash
    note_labels: dict[str, dict] = raw_labels
else:
    note_labels = {}

all_note_hashes = list(note_hash_to_ids.keys())
uncached = [nh for nh in all_note_hashes if nh not in note_labels]
print(f"  Do przetworzenia (bez cache): {len(uncached)}")

for batch_start in tqdm(range(0, len(uncached), BATCH_SIZE), desc="LLM batche"):
    batch_nhs = uncached[batch_start : batch_start + BATCH_SIZE]
    entries = [
        {"idx": i, "note": note_hash_to_note[nh]}
        for i, nh in enumerate(batch_nhs)
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(entries, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=6000,  # ~60 tokenów na wpis × 100 wpisów (z zapasem)
    )

    result_text = response.choices[0].message.content.strip()
    # Usuń ewentualne ```json ... ``` owijki
    result_text = re.sub(r"^```[a-z]*\s*", "", result_text)
    result_text = re.sub(r"\s*```$", "", result_text)

    metrics = extract_openai_usage_metrics(response)
    cost = calculate_usage_cost_usd(LLM_MODEL, metrics)
    append_usage_log(
        log_path,
        session_usage,
        action="llm_note_labeling",
        model=LLM_MODEL,
        usage_metrics=metrics,
        cost_usd=cost,
        payload={"batch_start": batch_start, "batch_size": len(batch_nhs), "result": result_text[:200]},
    )
    batch_num     = batch_start // BATCH_SIZE + 1
    total_batches = (len(uncached) + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"  [batch {batch_num}/{total_batches}] "
        f"tokens={metrics['input_tokens']}/{metrics['output_tokens']} "
        f"cost=${cost:.4f} | session=${session_usage['cost_usd']:.4f}"
    )

    # Parsuj i zapisz etykiety dla każdego note_hash w batchu
    match = re.search(r"\[.*\]", result_text, re.DOTALL)
    if not match:
        print(f"  WARN: brak tablicy JSON w odpowiedzi batch {batch_num}")
        continue
    try:
        labeled = json.loads(match.group(0))
        for item in labeled:
            idx = item.get("idx")
            if isinstance(idx, int) and 0 <= idx < len(batch_nhs):
                note_labels[batch_nhs[idx]] = {
                    "sentiment":       item.get("sentiment", "neutral"),
                    "sensor_mentions": item.get("sensor_mentions", []),
                    "coherent":        item.get("coherent", True),
                }
    except Exception as e:
        print(f"  WARN: parse error batch {batch_num}: {e} | raw: {result_text[:150]}")

# Zapisz cache po każdym batchu (nadpisuje)
LABELS_CACHE.write_text(json.dumps(note_labels, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  Zapisano {len(note_labels)} etykiet -> {LABELS_CACHE}")

# Podsumowanie etykiet
sentiments = {"ok": 0, "problem": 0, "neutral": 0, "unknown": 0}
incoherent_count = 0
for lb in note_labels.values():
    s = lb.get("sentiment", "unknown")
    sentiments[s] = sentiments.get(s, 0) + 1
    if not lb.get("coherent", True):
        incoherent_count += 1
print(f"  Sentymenty:  ok={sentiments['ok']}  problem={sentiments['problem']}  neutral={sentiments['neutral']}")
print(f"  Niespojne:   {incoherent_count}")

# ==============================================================================
# Faza 5: Programistyczne łączenie wyników
#
# Anomalie:
#   (A) dane złe (typ 1/4) → zawsze anomalia
#   (B) dane OK + notatka "problem" (typ 3) → anomalia notatki
#   (C) notatka incoherent → anomalia notatki (bez sensu = niepoprawna)
#   (D) notatka wymienia WYŁĄCZNIE sensory spoza sensor_type pliku → anomalia notatki
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 5: Lacznie wynikow per plik...")

anomaly_ids: list[str] = []
reasons_counter: dict[str, int] = {"data": 0, "note_problem": 0, "incoherent": 0, "sensor_mismatch": 0}

for file_id in sorted(file_data_hash.keys()):
    dh = file_data_hash[file_id]
    nh = file_note_hash[file_id]
    sensor_type = file_sensor_type[file_id]
    label = note_labels.get(nh, {})

    is_anomaly = False

    # (A) dane zle
    if dh in data_bad:
        is_anomaly = True
        reasons_counter["data"] += 1

    # Flagi z notatki
    sentiment  = label.get("sentiment", "neutral")
    coherent   = label.get("coherent", True)
    mentions   = set(label.get("sensor_mentions", []))
    actual_kws = set(sensor_type.lower().replace("/", " ").split())

    # (B) dane OK, notatka mowi o problemie
    if dh in data_ok and sentiment == "problem":
        is_anomaly = True
        reasons_counter["note_problem"] += 1

    # (C) notatka to bezsens
    if not coherent:
        is_anomaly = True
        reasons_counter["incoherent"] += 1

    # (D) notatka wymienia sensory, ale zadne nie sa w sensor_type tego pliku
    if mentions and not mentions.intersection(actual_kws):
        is_anomaly = True
        reasons_counter["sensor_mismatch"] += 1

    if is_anomaly:
        anomaly_ids.append(file_id)

anomaly_ids = sorted(set(anomaly_ids))

print(f"  Przyczyny anomalii (z duplikatami):")
for k, v in reasons_counter.items():
    print(f"    {k}: {v}")
print(f"  Lacznie plikow anomalii: {len(anomaly_ids)}")
print(f"  Przyklad: {anomaly_ids[:10]}")

# ==============================================================================
# Faza 6: Wysylanie do Centrali
# ==============================================================================

print("\n" + "=" * 60)
print("Faza 6: Wysylanie do Centrali...")

payload = {
    "apikey": AI_DEVS_API_KEY,
    "task": TASK,
    "answer": {"recheck": anomaly_ids},
}

result = send_verify(payload)
print(f"\n=== WYNIK ===")
print(json.dumps(result, ensure_ascii=False, indent=2))

print(f"\n=== PODSUMOWANIE ===")
print(f"  Anomalie wyslane:  {len(anomaly_ids)}")
print(f"  Tokeny:            in={session_usage['input_tokens']} out={session_usage['output_tokens']}")
print(f"  Koszt sesji:       ${session_usage['cost_usd']:.4f}")

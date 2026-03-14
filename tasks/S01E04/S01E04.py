# ==============================================================================
# S01E04 - System Przesyłek Konduktorskich
# Faza 1: Pobranie i organizacja danych
# ==============================================================================

import os
import requests
import re
from pathlib import Path
from datetime import date

from tqdm import tqdm

# Konfiguracja ścieżek
BASE_URL = "https://hub.ag3nts.org/dane/doc/"
DATA_DIR = Path("tasks/S01E04/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pobranie index.md
index_url = f"{BASE_URL}index.md"
response = requests.get(index_url)
response.raise_for_status()
index_content = response.text

with open(DATA_DIR / "index.md", "w", encoding="utf-8") as f:
    f.write(index_content)

print("Pobrano index.md")

# Wyciągnięcie linków do załączników (szukamy wzorca [include file="..."])
attachments = re.findall(r'\[include file="(.*?)"\]', index_content)
print("Znalezione załączniki:", attachments)

# Pobranie załączników
for attachment in tqdm(attachments, desc="Pobieranie załączników", total=len(attachments)):
    att_url = f"{BASE_URL}{attachment}"
    att_response = requests.get(att_url)
    att_response.raise_for_status()
    
    # Zapis binarny dla obrazków, tekstowy dla reszty
    mode = "wb" if attachment.endswith(".png") else "w"
    encoding = None if attachment.endswith(".png") else "utf-8"
    
    file_path = DATA_DIR / attachment
    with open(file_path, mode, encoding=encoding) as f:
        if mode == "wb":
            f.write(att_response.content)
        else:
            f.write(att_response.text)

    print(f"Pobrano {attachment}")

# ------------------------------------------------------------------------------
# Faza 2: Analiza danych i wyznaczenie parametrów przesyłki (Analiza obrazu)
# ------------------------------------------------------------------------------

import base64
import json
import urllib.request

# Odczytanie tekstu z obrazka z zablokowanymi trasami przy użyciu OpenAI Vision
image_path = DATA_DIR / "trasy-wylaczone.png"

with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

api_key = os.environ["OPENAI_API_KEY"]

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Wypisz z tego obrazka wszystkie informacje o zamkniętych trasach. Podaj kody tras i miasta, które łączą."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

req = urllib.request.Request("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload).encode('utf-8'))
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode('utf-8'))
    print("Zablokowane trasy z obrazka:")
    blocked_routes_text = result['choices'][0]['message']['content']
    print(blocked_routes_text)
with open(DATA_DIR / "trasy-wylaczane.txt", "w", encoding="utf-8") as f:
    f.write(blocked_routes_text)

# ------------------------------------------------------------------------------
# Faza 3: Wyznaczenie trasy i kategorii (Logika)
# ------------------------------------------------------------------------------

shipment_sender_id = "450202122"
shipment_origin = "Gdańsk"
shipment_destination = "Żarnowiec"
shipment_weight_kg = 2800
shipment_description = "kasety z paliwem do reaktora"

# Trasa do Żarnowca z mapy i obrazu tras wyłączonych.
# Zgodnie z dokumentacją trasy do Żarnowca mogą być użyte tylko dla kat. A/B.
route_code = "X-01"
shipment_category = "A"

# WDP = liczba dodatkowych wagonów ponad standardowy udźwig 1000 kg.
standard_train_capacity_kg = 1000
extra_wagon_capacity_kg = 500
wdp_count = max(0, (shipment_weight_kg - standard_train_capacity_kg + extra_wagon_capacity_kg - 1) // extra_wagon_capacity_kg)

# Kategorie A i B są finansowane przez System (0 PP).
payment_pp = 0

print("Wyznaczone parametry:")
print(f"Trasa: {route_code}")
print(f"Kategoria: {shipment_category}")
print(f"WDP: {wdp_count}")
print(f"Kwota do zapłaty: {payment_pp} PP")

# ------------------------------------------------------------------------------
# Faza 4: Przygotowanie deklaracji w formacie z załącznika E
# ------------------------------------------------------------------------------

declaration_text = (
    "SYSTEM PRZESYŁEK KONDUKTORSKICH - DEKLARACJA ZAWARTOŚCI\n"
    "======================================================\n"
    f"DATA: {date.today().isoformat()}\n"
    f"PUNKT NADAWCZY: {shipment_origin}\n"
    "------------------------------------------------------\n"
    f"NADAWCA: {shipment_sender_id}\n"
    f"PUNKT DOCELOWY: {shipment_destination}\n"
    f"TRASA: {route_code}\n"
    "------------------------------------------------------\n"
    f"KATEGORIA PRZESYŁKI: {shipment_category}\n"
    "------------------------------------------------------\n"
    f"OPIS ZAWARTOŚCI (max 200 znaków): {shipment_description}\n"
    "------------------------------------------------------\n"
    f"DEKLAROWANA MASA (kg): {shipment_weight_kg}\n"
    "------------------------------------------------------\n"
    f"WDP: {wdp_count}\n"
    "------------------------------------------------------\n"
    "UWAGI SPECJALNE: \n"
    "------------------------------------------------------\n"
    f"KWOTA DO ZAPŁATY: {payment_pp} PP\n"
    "------------------------------------------------------\n"
    "OŚWIADCZAM, ŻE PODANE INFORMACJE SĄ PRAWDZIWE.\n"
    "BIORĘ NA SIEBIE KONSEKWENCJĘ ZA FAŁSZYWE OŚWIADCZENIE.\n"
    "======================================================"
)

print("\nDeklaracja do wysłania:\n")
print(declaration_text)

with open(DATA_DIR / "declaration.txt", "w", encoding="utf-8") as f:
    f.write(declaration_text)

# ------------------------------------------------------------------------------
# Faza 5: Wysyłka do /verify
# ------------------------------------------------------------------------------

from tasks.commons.task_handler import AI_DEVS_API_KEY, send_verify

verify_payload = {
    "apikey": AI_DEVS_API_KEY,
    "task": "sendit",
    "answer": {
        "declaration": declaration_text,
    },
}

verify_response = send_verify(verify_payload)
print("\nOdpowiedź /verify:")
print(json.dumps(verify_response, ensure_ascii=False, indent=2))
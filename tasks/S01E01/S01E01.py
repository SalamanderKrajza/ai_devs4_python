import json
import os
import unicodedata
from pathlib import Path
from io import StringIO

import datetime
import pandas as pd
import requests
from google import genai
from google.genai import types

from tasks.commons.task_handler import ai_devs_key, send_verify


# --------------------------------------------------------------
# Validate environment
# --------------------------------------------------------------
gemini_api_key = os.environ["GEMINI_API_KEY"]


# --------------------------------------------------------------
# Get task data
# --------------------------------------------------------------
people_url = f"https://hub.ag3nts.org/data/{ai_devs_key}/people.csv"
people_response = requests.get(people_url, timeout=30)
people_response.raise_for_status()
df_people = pd.read_csv(StringIO(people_response.text))
df_people["birthDate"] = pd.to_datetime(df_people["birthDate"]).dt.date


# --------------------------------------------------------------
# Save data to file
# --------------------------------------------------------------
data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
df_people.to_csv(data_dir / "people.csv", index=False)


# --------------------------------------------------------------
# Filter candidates
# --------------------------------------------------------------
"""
Task condition:

→ są mężczyznami, którzy teraz w 2026 roku mają między 20, a 40 lat
→ urodzonych w Grudziądzu
→ pracują w branży transportowej
"""
gender = "M"
today = datetime.date.today()
current_year = today.year
min_birth_date = datetime.date(current_year - 40, today.month, today.day)
max_birth_date = datetime.date(current_year - 20, today.month, today.day)
location = "grudziądz"

df_candidates = df_people.query(
    "birthDate.between(@min_birth_date, @max_birth_date) "
    "and gender == @gender "
    "and birthPlace.str.lower() == @location"
)


# --------------------------------------------------------------
# Tag job descriptions with LLM (Structured Output)
# --------------------------------------------------------------
ALLOWED_TAGS = [
    "IT",
    "transport",
    "edukacja",
    "medycyna",
    "praca z ludźmi",
    "praca z pojazdami",
    "praca fizyczna",
]

TAG_DESCRIPTIONS = {
    "IT": "software, systemy, programowanie, infrastruktura IT, dane",
    "transport": "logistyka, przewozy, spedycja, transport drogowy/kolejowy/morski",
    "edukacja": "nauczanie, szkolenia, dydaktyka, praca w szkole/uczelni",
    "medycyna": "opieka zdrowotna, leczenie, diagnostyka, medyczne usługi",
    "praca z ludźmi": "obsługa klienta, HR, usługi społeczne, bezpośrednia praca z ludźmi",
    "praca z pojazdami": "kierowcy, operatorzy pojazdów, mechanicy pojazdów",
    "praca fizyczna": "praca manualna, produkcja, budowa, magazyn",
}

job_items = [
    {"id": idx, "job": row["job"]}
    for idx, row in df_candidates.reset_index(drop=True).iterrows()
]
tag_list = "\n".join(f"- {tag}: {desc}" for tag, desc in TAG_DESCRIPTIONS.items())
system_text = (
    "You are a classifier that assigns job descriptions to tags. "
    "Return only tags from the allowed list."
)
user_text = (
    "Assign tags for each job description. Use only the allowed tags below. "
    "Multiple tags are allowed.\n\n"
    f"Allowed tags:\n{tag_list}\n\n"
    "Jobs (JSON):\n"
    f"{json.dumps(job_items, ensure_ascii=False)}"
)

schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "enum": ALLOWED_TAGS},
                    },
                },
                "required": ["id", "tags"],
            },
        }
    },
    "required": ["items"],
}

gemini_model = "gemini-2.5-flash"
client = genai.Client(api_key=gemini_api_key)
llm_response = client.models.generate_content(
    model=gemini_model,
    contents=user_text,
    config=types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=system_text,
        response_mime_type="application/json",
        response_schema=schema,
    ),
)

structured_content = llm_response.text
tagged_items = json.loads(structured_content)["items"]
tags_by_id = {item["id"]: item["tags"] for item in tagged_items}


# --------------------------------------------------------------
# Build answer
# --------------------------------------------------------------
answers = []
for idx, row in df_candidates.reset_index(drop=True).iterrows():
    tags = tags_by_id.get(idx, [])
    if "transport" not in tags:
        continue

    answers.append(
        {
            "name": row["name"],
            "surname": row["surname"],
            "gender": row["gender"],
            "born": int(row["birthDate"].year),
            "city": row["birthPlace"],
            "tags": tags,
        }
    )

verify_payload = {"apikey": ai_devs_key, "task": "people", "answer": answers}


# --------------------------------------------------------------
# Save suspects for next task
# --------------------------------------------------------------
# Json format:
suspects_payload = [
    {
        "name": item["name"],
        "surname": item["surname"],
        "birthYear": item["born"],
    }
    for item in answers
]

suspects_path_json = data_dir / "suspects.json"
suspects_path_json.write_text(
    json.dumps(suspects_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# Csv format:
# suspects_path_csv = data_dir / "suspects.csv"
# pd.DataFrame(suspects_payload).to_csv(suspects_path_csv, index=False) 


# --------------------------------------------------------------
# Send answer
# --------------------------------------------------------------
result = send_verify(verify_payload)
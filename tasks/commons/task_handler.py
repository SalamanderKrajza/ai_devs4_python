# Get task data
import os
from dotenv import load_dotenv
import requests

load_dotenv()

AI_DEVS_API_KEY = os.environ["AI_DEVS_API_KEY"]

def send_verify(payload: dict) -> dict:
    url = "https://hub.ag3nts.org/verify"
    response = requests.post(url, json=payload, timeout=30)
    print("Status: ", response.status_code)
    print(response.json())
    response.raise_for_status()
    return response.json()
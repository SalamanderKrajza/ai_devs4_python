import os
import requests
import uuid
from dotenv import load_dotenv

# ==============================================================================
# 1. Setup
# ==============================================================================
load_dotenv()
AI_DEVS_API_KEY = os.getenv("AI_DEVS_API_KEY")

if not AI_DEVS_API_KEY:
    raise ValueError("Missing AI_DEVS_API_KEY in .env file")

# ==============================================================================
# 2. Local Testing (Optional)
# ==============================================================================
# Run this block to test your local server before exposing it via ngrok
def test_local_server(local_url: str = "http://localhost:3000/"):
    payload = {
        "sessionID": "test-session-1",
        "msg": "Cześć, sprawdź proszę paczkę PKG12345678"
    }
    print(f"Sending test message to local server: {payload['msg']}")
    try:
        response = requests.post(local_url, json=payload)
        print("Local server response:", response.json())
    except Exception as e:
        print("Error connecting to local server. Is it running? Error:", e)

# test_local_server()

# ==============================================================================
# 3. Task Verification
# ==============================================================================
# 1. Start the API server in terminal: `poetry run python tasks/S01E03/S01E03_api.py`
# 2. Start ngrok in another terminal: `ngrok http 3000`
# 3. Paste the ngrok URL below and run this block
NGROK_URL = "https://1081-34-118-42-165.ngrok-free.app "

def verify_task():
    if "YOUR-NGROK-URL" in NGROK_URL:
        print("Please update NGROK_URL before verifying.")
        return

    # Ensure URL has no trailing slash, or has it if needed. Let's try without first, 
    # but the hub might append paths.
    url = NGROK_URL.rstrip("/")
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"

    payload = {
        "apikey": AI_DEVS_API_KEY,
        "task": "proxy",
        "answer": {
            "url": url,
            "sessionID": session_id
        }
    }
    
    print(f"Sending verification request to hub.ag3nts.org for URL: {url} with sessionID: {session_id}")
    response = requests.post("https://hub.ag3nts.org/verify", json=payload)
    print("Verification response:")
    print(response.text)

# Uncomment to verify
# verify_task()

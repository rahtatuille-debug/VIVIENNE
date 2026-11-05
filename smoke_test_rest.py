"""Fallback smoke test using only the standard library (no pip packages).

Use this only if `pip install -r requirements.txt` fails on your macOS.
Run: GEMINI_API_KEY=your-key python3 smoke_test_rest.py
"""
import json
import os
import sys
import urllib.error
import urllib.request

key = os.getenv("GEMINI_API_KEY")
if not key:
    sys.exit("FAIL: set GEMINI_API_KEY first, e.g. GEMINI_API_KEY=abc python3 smoke_test_rest.py")
model = os.getenv("VIVIENNE_MODEL", "gemini-3.8-flash")

req = urllib.request.Request(
    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    data=json.dumps({"contents": [{"parts": [{"text": "Say hello in five words."}]}]}).encode(),
    headers={"Content-Type": "application/json", "x-goog-api-key": key},
)
try:
    with urllib.request.urlopen(req, timeout=25) as resp:
        body = json.load(resp)
except urllib.error.HTTPError as e:
    sys.exit(f"FAIL: HTTP {e.code}: {e.read().decode()[:300]}")
except urllib.error.URLError as e:
    sys.exit(f"FAIL: {e.reason}\nIf this mentions CERTIFICATE_VERIFY_FAILED, run the "
             "'Install Certificates.command' file in /Applications/Python 3.x/")

print(f"Model  : {model}")
print("Gemini :", body["candidates"][0]["content"]["parts"][0]["text"].strip())
print("PASS")

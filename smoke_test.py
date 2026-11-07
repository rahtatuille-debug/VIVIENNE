"""Phase 0 smoke test: print your machine facts, then get one reply from Gemini."""
import os
import platform
import subprocess
import sys


def machine_facts():
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        ram = f"{int(out.stdout.strip()) / 1024 ** 3:.0f} GB"
    except (OSError, ValueError):
        ram = "unknown"
    print(f"macOS  : {platform.mac_ver()[0] or 'not macOS'} ({platform.machine()})")
    print(f"Python : {platform.python_version()}")
    print(f"RAM    : {ram}")
    if sys.version_info < (3, 9):
        sys.exit("FAIL: google-genai needs Python 3.9+. Install a newer Python from python.org.")


def main():
    machine_facts()

    try:
        from dotenv import load_dotenv
        from google import genai
        from google.genai import errors, types
    except ImportError as e:
        sys.exit(f"FAIL: {e}\nFix: pip uninstall -y google-generativeai && "
                 "pip install --prefer-binary -r requirements.txt")

    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("FAIL: GEMINI_API_KEY missing. Run: cp .env.example .env  then paste your key.")
    model = os.getenv("VIVIENNE_MODEL", "gemini-3.5-flash").split(",")[0].strip()

    try:
        client = genai.Client(http_options=types.HttpOptions(timeout=25_000))  # reads GEMINI_API_KEY
        r = client.models.generate_content(model=model, contents="Say hello in five words.")
    except errors.APIError as e:
        msg = str(e.message or e)
        if e.code == 429:
            sys.exit("FAIL: 429 free-tier rate limit. Wait a minute and retry.")
        if e.code == 404:
            sys.exit(f"FAIL: model '{model}' not found. Copy the current Flash model name "
                     "from aistudio.google.com into VIVIENNE_MODEL in .env.")
        if e.code in (500, 502, 503, 504):
            sys.exit("FAIL: Google's servers are overloaded right now (not your fault). Retry in a few minutes.")
        if "API key" in msg or e.code in (401, 403):
            sys.exit("FAIL: Google rejected your API key. Recreate it at "
                     "aistudio.google.com/apikey and update .env.")
        sys.exit(f"FAIL: Gemini error {e.code}: {msg[:300]}")
    except Exception as e:
        text = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in text:
            sys.exit("FAIL: SSL certificates missing. Run the 'Install Certificates.command' "
                     "file in your /Applications/Python 3.x/ folder, then retry.")
        sys.exit(f"FAIL: {type(e).__name__}: {text[:300]}\nAre you online?")

    print(f"Model  : {model}")
    print(f"Gemini : {(r.text or '').strip()}")
    print("PASS")


if __name__ == "__main__":
    main()

"""The only file that talks to Gemini. Swap ask_brain()'s body to change provider."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
MODEL = os.getenv("VIVIENNE_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are Vivienne, a concise, composed personal assistant with dry wit. "
    "The user may call you Vivienne or Vivi. "
    "Replies will be spoken aloud, so use 1 to 3 short sentences, no markdown, no lists. "
    "Use your tools whenever they help. Never claim you did something unless a tool "
    "confirmed it. If a request is ambiguous, ask one short question. If you don't know, say so."
)

_client = None  # must stay referenced, or the SDK closes its connection
_chat = None


class BrainSetupError(Exception):
    """Raised when the brain can't start, for example a missing API key."""


def init_brain(tools):
    """Create the chat session. `tools` is a list of plain Python functions."""
    global _client, _chat
    if not os.getenv("GEMINI_API_KEY"):
        raise BrainSetupError(
            "GEMINI_API_KEY is missing. Run: cp .env.example .env  then paste your key into .env"
        )
    _client = genai.Client()  # reads GEMINI_API_KEY from the environment
    _chat = _client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=tools),
    )


def ask_brain(text: str) -> str:
    """The single entry point. Swap the body later to use Claude or Groq."""
    if _chat is None:
        return "My brain isn't switched on yet. Call init_brain() first."
    try:
        return _chat.send_message(text).text or "I have nothing to say to that."
    except errors.APIError as e:
        msg = str(e.message or e)
        if e.code == 429 or "RESOURCE_EXHAUSTED" in msg:
            return "I've hit the free-tier rate limit. Give me a minute."
        if e.code == 404:
            return f"Google doesn't recognise the model '{MODEL}'. Update VIVIENNE_MODEL in .env."
        if "API key" in msg:
            return "Google rejected my API key. Check GEMINI_API_KEY in .env."
        return f"Gemini returned an error ({e.code}): {msg[:200]}"
    except Exception as e:
        text_err = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in text_err:
            return "My SSL certificates are broken, so I can't reach Google."
        if any(w in type(e).__name__ for w in ("Connect", "Timeout", "Network")):
            return "I can't reach the internet right now. Check your Wi-Fi."
        return f"I couldn't reach my brain: {text_err[:200]}"

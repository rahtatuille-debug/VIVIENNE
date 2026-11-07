"""The only file that talks to Gemini. Swap ask_brain()'s body to change provider."""
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Models to try, in order. If one is busy, out of quota or retired, Vivienne moves to the next.
DEFAULT_MODELS = "gemini-3.5-flash,gemini-3.6-flash,gemini-3.8-flash,gemini-3.5-flash-lite"
MODELS = [m.strip() for m in os.getenv("VIVIENNE_MODEL", DEFAULT_MODELS).split(",") if m.strip()]

# Give up on one model after 12 seconds. No retries on the same model:
# moving to the next model is faster than waiting for a busy one.
HTTP_OPTIONS = types.HttpOptions(
    timeout=12_000,
    retry_options=types.HttpRetryOptions(attempts=1),
)

# Errors that mean "this model can't answer right now, try another one".
FALLBACK_CODES = {404, 429, 500, 502, 503, 504}

SYSTEM_PROMPT = (
    "You are Vivienne, a concise, composed personal assistant with dry wit. "
    "The user may call you Vivienne or Vivi. "
    "Replies will be spoken aloud, so use 1 to 3 short sentences, no markdown, no lists. "
    "Use your tools whenever they help. Never claim you did something unless a tool "
    "confirmed it. If a request is ambiguous, ask one short question. If you don't know, say so."
)

_client = None  # must stay referenced, or the SDK closes its connection
_config = None
_chat = None
_model_index = 0  # which model in MODELS is currently in use


class BrainSetupError(Exception):
    """Raised when the brain can't start, for example a missing API key."""


def init_brain(tools):
    """Create the chat session. `tools` is a list of plain Python functions."""
    global _client, _config
    if not os.getenv("GEMINI_API_KEY"):
        raise BrainSetupError(
            "GEMINI_API_KEY is missing. Run: cp .env.example .env  then paste your key into .env"
        )
    if not MODELS:
        raise BrainSetupError("VIVIENNE_MODEL in .env is empty. Put at least one model name in it.")
    _client = genai.Client(http_options=HTTP_OPTIONS)  # reads GEMINI_API_KEY
    _config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=tools)
    _start_chat(0)


def current_model() -> str:
    """The model Vivienne is using right now."""
    return MODELS[_model_index]


def _start_chat(index, history=None):
    """Start a chat on MODELS[index], carrying over the conversation so far."""
    global _chat, _model_index
    _model_index = index
    _chat = _client.chats.create(model=MODELS[index], config=_config, history=history or [])


def _can_fall_back(error) -> bool:
    """True if another model might succeed where this one failed."""
    if isinstance(error, errors.APIError):
        return error.code in FALLBACK_CODES
    return "Timeout" in type(error).__name__


def ask_brain(text: str) -> str:
    """The single entry point. Swap the body later to use Claude or Groq."""
    if _chat is None:
        return "My brain isn't switched on yet. Call init_brain() first."

    # A clean copy of the conversation, so a model that fails halfway through
    # a reply doesn't leave a half-finished turn for the next model.
    # list() makes a real copy; without it the failing chat keeps writing into it.
    history_before = list(_chat.get_history(curated=True))
    started_on = MODELS[_model_index]
    # Each model once, starting with the one in use: e.g. [B, C, A] if B is current.
    order = MODELS[_model_index:] + MODELS[:_model_index]
    failures = []

    for i, model in enumerate(order):
        if i > 0:
            if order[i - 1] in MODELS:  # a dropped model already said why
                print(f"  ({order[i - 1]} unavailable, trying {model})", file=sys.stderr)
            _start_chat(MODELS.index(model), history=history_before)
        try:
            return _chat.send_message(text).text or "I have nothing to say to that."
        except Exception as e:
            failures.append(e)
            if _is_missing_model(e) and len(MODELS) > 1:
                # Retired or misspelt: stop trying it for the rest of this session.
                MODELS.remove(model)
                print(f"  ({model} doesn't exist, skipping it from now on)", file=sys.stderr)
            if not _can_fall_back(e):
                break

    # Every model failed (or the error wasn't one another model could fix).
    # Next message starts again from the same model, with the clean conversation.
    back_to = MODELS.index(started_on) if started_on in MODELS else 0
    _start_chat(back_to, history=history_before)
    return _explain(_most_useful(failures))


def _is_missing_model(error) -> bool:
    return isinstance(error, errors.APIError) and error.code == 404


def _most_useful(failures):
    """Pick the error that best explains what went wrong.

    A wrong key beats "busy", which beats "too slow", which beats "doesn't exist".
    Otherwise one misspelt model name would hide that Google is just busy.
    """
    def rank(error):
        if not _can_fall_back(error):
            return 0  # e.g. a wrong key: the real problem, whatever else happened
        if isinstance(error, errors.APIError):
            return {429: 1, 404: 4}.get(error.code, 2)
        return 3  # timeout
    return min(failures, key=rank)


def _explain(error) -> str:
    """Turn an error into a short sentence Vivienne can say."""
    if isinstance(error, errors.APIError):
        msg = str(error.message or error)
        if error.code == 429 or "RESOURCE_EXHAUSTED" in msg:
            return "I've hit the free-tier limit on all my models. Give me a minute."
        if error.code == 404:
            return "Google doesn't recognise my model names. Update VIVIENNE_MODEL in .env."
        if error.code in (500, 502, 503, 504):
            return "Google's servers are busy right now. Try again in a minute."
        if "API key" in msg:
            return "Google rejected my API key. Check GEMINI_API_KEY in .env."
        return f"Gemini returned an error ({error.code}): {msg[:200]}"
    text_err = str(error)
    if "CERTIFICATE_VERIFY_FAILED" in text_err:
        return "My SSL certificates are broken, so I can't reach Google."
    name = type(error).__name__
    if "Connect" in name or "Network" in name:
        return "I can't reach the internet right now. Check your Wi-Fi."
    if "Timeout" in name:
        return "Gemini took too long to answer. Try again."
    return f"I couldn't reach my brain: {text_err[:200]}"

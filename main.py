"""Text chat with Vivienne. Run: python main.py"""
import sys

from brain import BrainSetupError, ask_brain, init_brain
from tools.basic import get_time, open_app, take_note


def main():
    try:
        init_brain([get_time, take_note, open_app])
    except BrainSetupError as e:
        sys.exit(str(e))

    print("Vivienne online. Type 'quit' to exit.")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"quit", "exit"}:
            break
        if user:
            print("Vivienne:", ask_brain(user))


if __name__ == "__main__":
    main()

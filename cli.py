"""Command line interface for the Solera agent.

    python -m solera_agent.cli --check
    python -m solera_agent.cli "what was revenue last month?"
    python -m solera_agent.cli            # interactive
"""

from __future__ import annotations

import argparse
import sys

import db
from agent import build_agent


def _print_stream(agent, question: str, show_sql: bool) -> None:
    state = {"messages": [{"role": "user", "content": question}]}

    for chunk in agent.stream(state, stream_mode="values"):
        message = chunk["messages"][-1]

        if getattr(message, "tool_calls", None) and show_sql:
            for call in message.tool_calls:
                query = call.get("args", {}).get("query", "")
                if query:
                    print("\n\033[2m--- sql ---")
                    print(query.strip())
                    print("-----------\033[0m")

        elif getattr(message, "type", None) == "tool" and show_sql:
            preview = message.content
            if len(preview) > 600:
                preview = preview[:600] + " ..."
            print(f"\033[2m{preview}\033[0m")

    print("\n" + chunk["messages"][-1].content + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about Solera data.")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive mode")
    parser.add_argument("--check", action="store_true", help="run the database health check and exit")
    parser.add_argument("--quiet", action="store_true", help="hide the generated SQL")
    parser.add_argument("--model", default=None, help="override the model, e.g. claude-sonnet-4-6")
    args = parser.parse_args()

    if args.check:
        print(db.health_check())
        return 0

    agent = build_agent(args.model)
    show_sql = not args.quiet

    if args.question:
        _print_stream(agent, " ".join(args.question), show_sql)
        return 0

    print("Solera analytics agent. Ctrl-C or an empty line to exit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        try:
            _print_stream(agent, question, show_sql)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
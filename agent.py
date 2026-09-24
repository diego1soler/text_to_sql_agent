""" SOLERA TEXT-TO-SQL AGENT
This is a LangGraph agent that answers questions about Solera's business by
querying its Postgres database, converting natural language to SQL. It is a read-only agent.

Design notes:

* No list_tables / get_schema tools. The schema is small and fixed, so it is
  injected into the system prompt from semantic_layer.yaml. That removes two
  LLM round-trips per question and lets the prompt carry business meaning.

* Validation lives inside the tool rather than in a separate LLM "query
  checker" node. A deterministic check is faster, free, and it is covered against hallucination. 
  When it rejects a query it returns a specific message the model
  can act on, which is what drives the self-correction loop.

* A retry cap. Without one, a model that keeps writing the same broken query
  will loop until it exhausts the context window or model's budget.
"""

from __future__ import annotations

import os
import pathlib
from typing import Literal

import yaml
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import db
from validator import ValidationError, validate

from dotenv import load_dotenv
load_dotenv()

MAX_SQL_ATTEMPTS = 5

_HERE = pathlib.Path(__file__).parent


def load_semantic_layer() -> str:
    """Read semantic_layer.yaml and flatten it into prompt text."""
    raw = (_HERE / "semantic_layer.yaml").read_text()
    yaml.safe_load(raw)
    return raw


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQL query to the Solera Postgres database and return
    the results as a table.

    Send exactly one SELECT statement. The database is read-only; INSERT,
    UPDATE, DELETE and DDL will be rejected. Results are capped at 1000 rows.

    If the query is rejected or errors, read the message, fix the query, and
    call this tool again.
    """
    try:
        validated = validate(query)
    except ValidationError as e:
        return f"The query has been rejected: {e}"

    try:
        result = db.run_query(validated.sql)
    except Exception as e:
        # Postgres error are passed through so the model can correct a bad column name or type cast on its own.
        return f"Query failed: {type(e).__name__}: {e}"

    prefix = ""
    if validated.limit_applied:
        prefix = "Note: a LIMIT 1000 was added automatically.\n"
    return prefix + result.to_markdown()


SYSTEM_PROMPT = """You are the analytics agent for SOLERA, the main digital transaction platform for a telecom company. 
You answer questions about the business by querying its Postgres database. 
These questions most likely involve aggregating metrics over time, comparing metrics across dimensions, or 
finding the top/bottom N of a metric.

These are the steps to follow when answering a question:
- Write one SQL query, run it with the run_sql tool, and answer from the result.
- Prefer a single well-constructed query over several exploratory ones.
- If a query is rejected or errors, read the message and fix it. Do not repeat the same query.
- Answer in prose, with the numbers. Do not paste the SQL back unless asked, 
but do state any assumption you made about the question.
- If the data cannot answer the question, say so plainly. Never estimate a number that you could not retrieve.
- As you are used by data analysts, they may ask the query back, so you should give the SQL query back IF and ONLY IF asked.. 
- If you are not able to answer the question, always suggest to ask back to the SOLERA data analytics team for further assistance.

The business context below defines what the metrics mean. It overrides your 
intuitions about what a column name implies, namely, read the metric definitions before writing aggregates.

--- BEGIN SEMANTIC LAYER ---
{semantic_layer}
--- END SEMANTIC LAYER ---
"""


def build_agent(model_name: str | None = None):
    """Compile and return the agent graph."""
    model_name = model_name or os.environ.get("SOLERA_MODEL", "claude-sonnet-4-6")
    model = init_chat_model(model_name, temperature=0)

    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(semantic_layer=load_semantic_layer()),
    }

    def generate(state: MessagesState):
        model_with_tools = model.bind_tools([run_sql])
        response = model_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["run_sql", "give_up", END]:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END

        attempts = sum(
            1 for m in state["messages"]
            if getattr(m, "type", None) == "tool" and getattr(m, "name", "") == "run_sql"
        )
        if attempts >= MAX_SQL_ATTEMPTS:
            return "give_up"
        return "run_sql"

    def give_up(state: MessagesState):
        """Cut the loop and make the model explain the failure rather than
        silently returning the last tool error."""
        note = {
            "role": "user",
            "content": (
                f"You have used all {MAX_SQL_ATTEMPTS} query attempts. Stop "
                "querying. Explain in one short paragraph what you were trying "
                "to find, what went wrong, and what you would need to answer it."
            ),
        }
        response = model.invoke([system_message] + state["messages"] + [note])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("generate", generate)
    builder.add_node("run_sql", ToolNode([run_sql], name="run_sql"))
    builder.add_node("give_up", give_up)

    builder.add_edge(START, "generate")
    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("run_sql", "generate")
    builder.add_edge("give_up", END)

    return builder.compile()


def ask(question: str, model_name: str | None = None) -> str:
    """Returns the final answer text."""
    agent = build_agent(model_name)
    final = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return final["messages"][-1].content


if __name__ == "__main__":
    # Quick manual test.
    #   python -m agent "how many transactions have been in the last month?"
    import sys

    question = " ".join(sys.argv[1:]) or "How many transactions are in the database?"

    agent = build_agent()
    print(f"Q: {question}\n")

    # stream=values gives the full message list after each node
    seen = 0
    final_state = None
    for state in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        final_state = state
        for msg in state["messages"][seen:]:
            role = getattr(msg, "type", "?")
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"[tool call] {call['name']}({call['args']})\n")
            if role == "tool":
                print(f"[tool result]\n{msg.content}\n")
            elif role == "ai" and msg.content:
                text = msg.content
                if isinstance(text, list):  # anthropic content blocks
                    text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
                print(f"[assistant]\n{text}\n")
        seen = len(state["messages"])

    attempts = sum(
        1 for m in final_state["messages"]
        if getattr(m, "type", None) == "tool" and getattr(m, "name", "") == "run_sql"
    )
    print(f"--- done ({attempts} run_sql call(s)) ---")
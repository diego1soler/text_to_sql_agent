"""Thin FastAPI wrapper around the agent.

    uvicorn api:app --reload

No auth, no rate limiting: this exposes an LLM key and a database connection,
so don't point a public deployment at it without adding both.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
from agent import build_agent

_agents: dict[str | None, object] = {}


def _agent_for(model: str | None):
    if model not in _agents:
        _agents[model] = build_agent(model)
    return _agents[model]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _agent_for(None)  # build the default agent eagerly so the first request isn't slow
    yield


app = FastAPI(title="Solera Text-to-SQL Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    model: str | None = None


class AskResponse(BaseModel):
    answer: str
    sql: list[str]
    run_sql_attempts: int


@app.get("/health")
def health() -> dict:
    return {"database": db.health_check()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    agent = _agent_for(request.model)
    state = agent.invoke({"messages": [{"role": "user", "content": request.question}]})

    sql: list[str] = []
    attempts = 0
    for message in state["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") == "run_sql":
                sql.append(call.get("args", {}).get("query", ""))
        if getattr(message, "type", None) == "tool" and getattr(message, "name", "") == "run_sql":
            attempts += 1

    return AskResponse(
        answer=state["messages"][-1].content,
        sql=sql,
        run_sql_attempts=attempts,
    )

"""Database access for the Solera agent.

Every query runs inside an explicitly read-only transaction with a statement
timeout, on a connection authenticated as `solera_agent`. 
That role already has no write privileges, so the read-only transaction is already safe, but it
costs nothing to add protection if someone points this at a connection string with more privileges than intended.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
import psycopg
from psycopg.rows import dict_row

MAX_ROWS = 1000
STATEMENT_TIMEOUT_MS = 20_000


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: int = 0

    def to_markdown(self, max_display: int = 50) -> str:
        """Compact rendering for the model. Wide or long results are trimmed
        the model needs enough to answer, not the whole result set."""
        if not self.rows:
            return "(0 rows)"

        shown = self.rows[:max_display]
        header = " | ".join(self.columns)
        sep = " | ".join("---" for _ in self.columns)
        body = "\n".join(
            " | ".join("" if r.get(c) is None else str(r.get(c)) for c in self.columns)
            for r in shown
        )
        out = f"{header}\n{sep}\n{body}"

        notes = [f"({self.row_count} rows, {self.elapsed_ms}ms)"]
        if len(self.rows) > max_display:
            notes.append(f"showing first {max_display}")
        if self.truncated:
            notes.append(
                f"result was capped at {MAX_ROWS} rows — aggregate in SQL rather "
                "than returning raw rows"
            )
        return out + "\n" + " ".join(notes)


def _dsn() -> str:
    dsn = os.environ.get("SOLERA_AGENT_DSN")
    if not dsn:
        raise RuntimeError(
            "SOLERA_AGENT_DSN is not set. It should be a Postgres connection string pointing at the Solera database."
        )
    if "postgres:" in dsn.split("@")[0] and "solera_agent" not in dsn:
        raise RuntimeError(
            "SOLERA_AGENT_DSN appears to use the postgres superuser role. FORBIDDEN. Use the solera_agent role instead."
        )
    return dsn


def run_query(sql: str, max_rows: int = MAX_ROWS) -> QueryResult:
    """Execute a validated SELECT and return at most max_rows rows."""
    started = time.perf_counter()

    with psycopg.connect(_dsn(), row_factory=dict_row, connect_timeout=10) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(sql)

            if cur.description is None:
                return QueryResult(elapsed_ms=int((time.perf_counter() - started) * 1000))

            columns = [d.name for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]

    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def health_check() -> str:
    """Confirm the connection works and is actually restricted."""
    checks = []

    result = run_query("select count(*) as n from solera_transactions_enriched")
    checks.append(f"read transactions: {result.rows[0]['n']} rows visible")

    if result.rows[0]["n"] == 0:
        checks.append(
            "WARNING: zero rows visible. The agent role is probably missing an "
            "RLS policy."
        )

    try:
        run_query("select count(*) from synthetic_user_traits")
        checks.append("FAIL: synthetic_user_traits is readable and should not be")
    except psycopg.Error:
        checks.append("blocked synthetic_user_traits: ok")

    try:
        run_query("create table _agent_probe (a int)")
        checks.append("FAIL: agent can create tables")
    except psycopg.Error:
        checks.append("blocked DDL: ok")

    return "\n".join(checks)


if __name__ == "__main__":
    print(health_check())
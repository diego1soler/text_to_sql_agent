# Solera Text-to-SQL Agent

A LangGraph agent that answers natural-language business questions by writing
and running its own read-only SQL against a Postgres database, self-correcting
when a query fails.

```
$ python agent.py "what was revenue last month, by currency?"

Q: what was revenue last month, by currency?

[tool call] run_sql({'query': "select currency, round(sum(amount), 2) as revenue from ..."})

[tool result]
currency | revenue
---      | ---
EUR      | 128430.55
USD      | 94210.10
(2 rows, 42ms)

[assistant]
Revenue last month was EUR 128,430.55 and USD 94,210.10. Kept by currency
since amounts aren't FX-converted in this dataset.

--- done (1 run_sql call(s)) ---
```

## Why this exists

Most text-to-SQL demos stop at "LLM writes a query." The interesting problems
are what happens around that: how does the model know what a metric like
*revenue* or *conversion rate* actually means in this business, what stops it
from running something destructive, and what happens when the query it wrote
is wrong. This project is a small, opinionated answer to those three
questions.

## Architecture

```
              ┌───────────┐   no tool call    ┌─────┐
   START ───▶ │ generate  │ ─────────────────▶ │ END │
              └───────────┘                    └─────┘
                 │      ▲
       tool call │      │ tool result
                 ▼      │
              ┌───────────┐
              │  run_sql  │
              └───────────┘

   generate ─▶ give_up ─▶ END      (after MAX_SQL_ATTEMPTS failed queries)
```

- **[agent.py](agent.py)** — the LangGraph graph. `generate` calls the model
  (Claude, via `init_chat_model`) with the system prompt and conversation so
  far; if it emits a `run_sql` tool call, `run_sql` executes it and the loop
  returns to `generate` so the model can read the result or fix its query.
  After 5 failed attempts, `give_up` short-circuits the loop and asks the
  model to explain what went wrong instead of retrying forever.
- **[validator.py](validator.py)** — parses model-generated SQL with
  [sqlglot](https://github.com/tobymao/sqlglot) (AST-based, not regex) and
  rejects anything that isn't a single read-only `SELECT`: no
  `INSERT`/`UPDATE`/`DELETE`/DDL, no blocked tables or schemas
  (`pg_catalog`, `information_schema`, a synthetic PII table), no
  file/connection-escaping functions (`pg_read_file`, `dblink`, `pg_sleep`,
  ...), and it injects a `LIMIT 1000` if the model didn't. A rejected query
  returns a specific, actionable message so the model can correct itself on
  the next turn.
- **[db.py](db.py)** — runs the validated query in an explicitly read-only
  transaction, on a connection authenticated as a Postgres role
  (`solera_agent`) that has no write grants at all. Includes a
  `health_check()` you can run standalone to confirm the role is actually
  restricted (blocked tables, blocked DDL) before wiring up the agent.
- **[api.py](api.py)** — thin FastAPI wrapper (`POST /ask`, `GET /health`)
  around the same `build_agent()` used by the CLI, for serving this over
  HTTP. No auth or rate limiting — see [Notes](#notes).
- **[semantic_layer.yaml](semantic_layer.yaml)** — the business context: table
  and column meanings, and metric definitions (e.g. *revenue* excludes
  refunds and pending payments; *orders* counts sessions, not line items).
  This is injected straight into the system prompt so the agent doesn't have
  to guess what a column implies, and so the schema doesn't cost a
  `list_tables` / `get_schema` round trip on every question.

### Three layers, each doing a different job

1. **Postgres role permissions** — the actual security boundary. Even if
   every layer above this had a bug, `solera_agent` cannot write or read
   restricted tables.
2. **The validator** — not primarily a security layer (layer 1 already
   handles that); its job is to turn a bad query into a *specific, correctable*
   error instead of an opaque Postgres permission failure.
3. **The system prompt / semantic layer** — steers the model toward correct
   *business logic* (what "revenue" means here), which no amount of SQL
   permissioning can enforce.

## Setup

```bash
git clone <this-repo>
cd textToSQLAgent
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the values below
```

`.env`:

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Claude API key. |
| `SOLERA_AGENT_DSN` | yes | Postgres connection string for a **read-only** role. Must not be a superuser DSN — [db.py](db.py) refuses to start if it looks like one. |
| `SOLERA_MODEL` | no | Defaults to `claude-sonnet-4-6`. |

Then sanity-check the database role before running the agent:

```bash
python db.py
```

## Run

```bash
python cli.py "how many transactions were there last week?"   # one-shot
python cli.py                                                  # interactive
python cli.py --check                                           # db.health_check() only
python cli.py --quiet "..."                                     # hide the generated SQL
```

[cli.py](cli.py) is the intended entry point. [agent.py](agent.py) can also be
run directly (`python agent.py "..."`) for a more verbose step-by-step trace
of the graph while developing.

## API

```bash
uvicorn api:app --reload
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "how many orders were completed in the last 90 days?"}'
# {"answer": "...", "sql": ["select count(distinct session_id) ..."], "run_sql_attempts": 1}

curl localhost:8000/health
```

Interactive docs at `localhost:8000/docs`.

## Evals

[evals.py](evals.py) is a ~20-case suite that checks the agent's *answers*,
not just that it runs:

```bash
python evals.py                    # full suite
python evals.py --list             # see all cases
python evals.py --category currency
python evals.py --repeat 3         # flag flaky cases (verdict changes across runs)
```

Three kinds of checks, each catching a different failure mode:
- **Hard** — re-runs the agent's SQL and diffs it against a hand-written
  ground-truth query. Gates the exit code.
- **Structural** — parses the agent's SQL with sqlglot and asserts it queried
  the right tables (e.g. conversion rate must come from `events`, not
  `solera_transactions`), independent of whether the number looked plausible.
- **Soft** — behavioural checks (did it state its assumption, name the right
  winner) that are reported but don't fail the build, since they're often a
  judgement call.

The case list deliberately includes paraphrases of the same question (to
separate "wrong wording" from "wrong understanding"), planted signals with a
known correct answer, and questions the data genuinely cannot answer — where
the only correct response is a refusal, not a plausible-looking guess.

## Notes

- The underlying data is synthetic and regenerated to the current date; there
  is no real user or transaction data behind this project.
- This is a demo/portfolio project, not a production service: it has no
  authentication or rate limiting, so don't point a public endpoint at it
  without adding both.

## License

MIT — see [LICENSE](LICENSE).

"""Validation of the model-generated SQL.

This runs before anything reaches the database. It is the second of three
layers, and is the only that inspects the query itself.

  1. Postgres role permissions  — the read-only solera_agent role is the security boundary
  2. This validator             — catches mistakes early with a useful message
  3. The system prompt          

The point of layer 2 is not primarily security; the read-only role already
handles that. It is that a rejected query produces a specific error the model
can correct on the next turn, instead of a Postgres permission error that
tells it nothing about what it did wrong.

Validation is AST-based, not regex.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

MAX_ROWS = 1000

# Statement types that must never appear anywhere in the tree, including inside
# CTEs and subqueries, using the SQLGLOT parser.
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Grant, exp.Merge,
)

# Tables and schemas the agent MUST NOT read even if a grant is accidentally added later.
BLOCKED_TABLES = {
    "synthetic_user_traits",
    "pg_shadow",
    "pg_authid",
    "pg_user",
}

BLOCKED_SCHEMAS = {"pg_catalog", "information_schema", "auth", "storage", "vault"}

# Functions that read files, open connections, or otherwise escape the query.
BLOCKED_FUNCTIONS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "lo_import", "lo_export",
    "dblink", "dblink_exec", "pg_sleep", "pg_terminate_backend", "pg_reload_conf",
    "copy",
}


class ValidationError(Exception):
    """Raised when generated SQL fails validation. The message is fed back to
    the model, so it should state what to do differently."""


@dataclass
class ValidatedQuery:
    sql: str
    limit_applied: bool


def validate(sql: str, max_rows: int = MAX_ROWS) -> ValidatedQuery:
    """Parse and check a single SELECT statement, returning normalised SQL.

    Raises ValidationError with a message intended for the model.
    """
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ValidationError("Empty query.")

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception as e:
        raise ValidationError(f"Could not parse as Postgres SQL: {e}") from e

    statements = [s for s in statements if s is not None]

    if len(statements) == 0:
        raise ValidationError("No statement found.")
    if len(statements) > 1:
        raise ValidationError(
            "Multiple statements found. Send exactly one SELECT statement, "
            "with no semicolon-separated follow-ups."
        )

    tree = statements[0]

    # The root must be a SELECT, or a WITH whose body is a SELECT.
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        raise ValidationError(
            f"Only SELECT queries are allowed; got {type(tree).__name__.upper()}. "
            "REMEMBER this database is read-only. Do not try any other statement type, even if you think it will work."
        )

    for node_type in FORBIDDEN_NODES:
        found = list(tree.find_all(node_type))
        if found:
            raise ValidationError(
                f"{node_type.__name__.upper()} is not allowed. Read-only SELECT queries only."
            )

    # Table and schema checks
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        schema = (table.db or "").lower()

        if schema in BLOCKED_SCHEMAS:
            raise ValidationError(
                f"Schema '{schema}' is not accessible. Query only the documented public tables."
            )
        if name in BLOCKED_TABLES:
            raise ValidationError(
                f"Table '{name}' is not accessible to this agent"
            )
        if name.startswith("pg_"):
            raise ValidationError(
                f"System catalog '{name}' is not accessible. YOU ARE NOT GRANTED."
            )

    # Function checks
    for func in tree.find_all(exp.Anonymous):
        fname = (func.this or "")
        if isinstance(fname, str) and fname.lower() in BLOCKED_FUNCTIONS:
            raise ValidationError(f"Function {fname}() is not allowed.")

    # Enforce a row limit. A missing LIMIT on a large table could blow the model's context window long before it becomes a database problem.
    limit_applied = False
    if isinstance(tree, exp.Select):
        existing = tree.args.get("limit")
        if existing is None:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
            limit_applied = True
        else:
            try:
                if int(existing.expression.this) > max_rows:
                    tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
                    limit_applied = True
            except (AttributeError, ValueError):
                pass
    elif isinstance(tree, exp.Union):
        # Wrap the union so the cap applies to the combined result.
        tree = exp.select("*").from_(tree.subquery(alias="u")).limit(max_rows)
        limit_applied = True

    return ValidatedQuery(sql=tree.sql(dialect="postgres", pretty=True),
                          limit_applied=limit_applied)

##Test code for validator.py
if __name__ == "__main__":
    ok = "select currency, sum(amount) from solera_transactions_enriched group by currency"
    bad = [
        "delete from solera_transactions",
        "select 1; select 2",
        "select * from synthetic_user_traits",
        "with x as (delete from users returning *) select * from x",
        "select * from pg_catalog.pg_roles",
    ]

    print(validate(ok).sql, "\n")
    for q in bad:
        try:
            validate(q)
            print(f"NOT CAUGHT: {q}")
        except ValidationError as e:
            print(f"caught: {q!r}\n   -> {e}")
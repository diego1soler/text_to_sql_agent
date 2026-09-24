"""Expanded evaluation suite.

Grading philosophy:

  HARD checks gate the exit code. They re-run the SQL the agent generated and
  compare its numbers against a hand-written ground-truth query. Arithmetic is
  either right or it isn't.

  SOFT checks are advisory and reported separately. They cover behaviour rather
  than arithmetic — did it state its assumption, did it group by currency, did
  it name the right winner. These are worth watching but a failure is a
  judgement call, not a bug, so they don't fail the build.

  STRUCTURAL checks use sqlglot to confirm the agent queried the right tables.
  Deterministic and independent of the numbers: a conversion rate computed from
  `solera_transactions` is wrong even if the figure happens to look plausible.

CONTAMINATION WARNING
Do not copy these questions into semantic_layer.yaml or the query library. If
you do, they pass and measure nothing — you have trained on the test set. When
a case fails, write the general RULE the failure revealed, then check that the
matching paraphrase case passes too. The paraphrase cases are the control group.

    python -m solera_agent.evals
    python -m solera_agent.evals --verbose
    python -m solera_agent.evals --repeat 3          # flakiness check
    python -m solera_agent.evals --category currency
    python -m solera_agent.evals --list
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

import sqlglot
from sqlglot import exp

import db
from agent import build_agent

TOLERANCE = 0.01


@dataclass
class Case:
    id: str
    question: str
    category: str
    note: str = ""

    # hard check: any numeric cell in the agent's result must match this
    truth_sql: str | None = None

    # structural check: the agent's SQL must reference all of these tables
    must_use_tables: list[str] = field(default_factory=list)

    # soft checks against the answer text
    must_mention: list[str] = field(default_factory=list)
    must_mention_any: list[str] = field(default_factory=list)

    # refusal check: the answer must contain no number at all
    forbid_number: bool = False

    # soft cases report but never fail the build
    soft: bool = False


CASES: list[Case] = [

    # ---------------------------------------------------------------- metrics
    Case(
        id="rev_30d_eur",
        category="metrics",
        question="How much completed revenue did we make in the last 30 days, in EUR?",
        truth_sql="""
            select round(sum(amount), 2) as v from solera_transactions_enriched
            where status = 'completed' and currency = 'EUR'
              and transaction_date >= now() - interval '30 days'
        """,
        note="Refunded and pending must be excluded.",
    ),
    Case(
        id="orders_90d",
        category="metrics",
        question="How many orders were completed in the last 90 days?",
        truth_sql="""
            select count(distinct session_id)::numeric as v
            from solera_transactions_enriched
            where status = 'completed' and transaction_date >= now() - interval '90 days'
        """,
        note="An order is a session. count(*) counts line items and is the trap.",
    ),
    Case(
        id="line_items_30d",
        category="metrics",
        question="How many individual line items were sold in the last 30 days?",
        truth_sql="""
            select count(*)::numeric as v from solera_transactions_enriched
            where status = 'completed' and transaction_date >= now() - interval '30 days'
        """,
        note="The inverse of orders_90d. Asking for items should NOT dedupe by session.",
    ),
    Case(
        id="aov_pro_eur",
        category="metrics",
        question="What is the average order value for pro tier users in EUR?",
        truth_sql="""
            with o as (
              select session_id, sum(amount) as order_value
              from solera_transactions_enriched
              where status = 'completed' and plan_tier = 'pro' and currency = 'EUR'
              group by session_id
            ) select round(avg(order_value), 2) as v from o
        """,
        note="Must aggregate to session level before averaging.",
    ),
    Case(
        id="refund_rate_90d",
        category="metrics",
        question="What percentage of transactions in the last 90 days were refunded?",
        truth_sql="""
            select round(100.0 * count(*) filter (where status = 'refunded') / count(*), 2) as v
            from solera_transactions
            where transaction_date >= now() - interval '90 days'
        """,
    ),

    # ------------------------------------------------------------ paraphrases
    # Same underlying question as a case above, worded differently. If the
    # original passes and the paraphrase fails, the fix was to the wording
    # rather than the understanding.
    Case(
        id="rev_paraphrase",
        category="paraphrase",
        question="How much money did we actually bring in over the past month in euros?",
        truth_sql="""
            select round(sum(amount), 2) as v from solera_transactions_enriched
            where status = 'completed' and currency = 'EUR'
              and transaction_date >= now() - interval '30 days'
        """,
        note="Control for rev_30d_eur. 'Actually brought in' should still exclude pending.",
    ),
    Case(
        id="aov_paraphrase",
        category="paraphrase",
        question="On average, how much does a pro subscriber spend each time they check out? Use euros.",
        truth_sql="""
            with o as (
              select session_id, sum(amount) as order_value
              from solera_transactions_enriched
              where status = 'completed' and plan_tier = 'pro' and currency = 'EUR'
              group by session_id
            ) select round(avg(order_value), 2) as v from o
        """,
        note="Control for aov_pro_eur. 'Each time they check out' is the session hint.",
    ),
    Case(
        id="orders_paraphrase",
        category="paraphrase",
        question="How many separate purchases have people made in the last three months?",
        truth_sql="""
            select count(distinct session_id)::numeric as v
            from solera_transactions_enriched
            where status = 'completed' and transaction_date >= now() - interval '90 days'
        """,
        note="Control for orders_90d.",
    ),

    # --------------------------------------------------------- funnel / events
    Case(
        id="conversion_rate",
        category="funnel",
        question="What is the overall session conversion rate, as a percentage?",
        truth_sql="""
            with s as (
              select session_id, bool_or(event_type = 'purchase') as converted
              from events group by session_id
            ) select round(100.0 * count(*) filter (where converted) / count(*), 2) as v from s
        """,
        must_use_tables=["events"],
        note="Transactions have no denominator. Must come from events.",
    ),
    Case(
        id="cart_abandon",
        category="funnel",
        question="How many sessions added something to the cart but never purchased?",
        truth_sql="""
            with s as (
              select session_id,
                     bool_or(event_type = 'add_to_cart') as carted,
                     bool_or(event_type = 'purchase') as bought
              from events group by session_id
            ) select count(*)::numeric as v from s where carted and not bought
        """,
        must_use_tables=["events"],
    ),
    Case(
        id="channel_conversion",
        category="funnel",
        question="Which acquisition channel has the best conversion rate?",
        truth_sql="""
            with s as (
              select e.session_id, min(u.acquisition_channel) as channel,
                     bool_or(e.event_type = 'purchase') as converted
              from events e join users u using (user_id) group by e.session_id
            )
            select round(100.0 * count(*) filter (where converted) / count(*), 2) as v
            from s group by channel order by v desc limit 1
        """,
        must_use_tables=["events", "users"],
        must_mention=["referral"],
        note="Planted signal: referral should win. Mention check is soft.",
    ),

    # -------------------------------------------------------- planted signals
    Case(
        id="worst_payment_method",
        category="signal",
        question="Which payment method has the highest failure rate, and what is that rate as a percentage?",
        truth_sql="""
            select round(100.0 * count(*) filter (where status = 'failed') / count(*), 2) as v
            from solera_transactions group by payment_method order by v desc limit 1
        """,
        must_mention=["bank"],
        note="bank_transfer is planted at ~13%.",
    ),
    Case(
        id="mobile_vs_desktop_aov",
        category="multi_step",
        question="Do mobile users spend less per order than desktop users? Answer in EUR.",
        truth_sql="""
            with o as (
              select session_id, device_type, sum(amount) as order_value
              from solera_transactions_enriched
              where status = 'completed' and currency = 'EUR'
              group by session_id, device_type
            )
            select round(avg(order_value), 2) as v from o where device_type = 'mobile'
        """,
        must_mention_any=["desktop"],
        note="Needs session aggregation, a split, and a comparison.",
    ),
    Case(
        id="top_merchant",
        category="multi_step",
        question="Which merchant generates the most completed revenue in EUR, and how much?",
        truth_sql="""
            select round(sum(t.amount), 2) as v
            from solera_transactions_enriched t
            where t.status = 'completed' and t.currency = 'EUR'
            group by t.merchant order by v desc limit 1
        """,
    ),

    # ------------------------------------------------------------- currency
    Case(
        id="total_revenue_all_currencies",
        category="currency",
        question="What is our total revenue, all time?",
        must_mention_any=["EUR", "GBP", "USD", "currenc"],
        soft=True,
        note="No FX conversion exists. A correct answer splits by currency rather "
             "than producing one meaningless total. Soft: wording varies.",
    ),
    Case(
        id="revenue_by_country",
        category="currency",
        question="Which country generates the most revenue?",
        must_mention_any=["currenc", "EUR", "GBP", "USD"],
        soft=True,
        note="US and GB rows are in different currencies. Ranking by raw amount "
             "across currencies is wrong and the answer should say so.",
    ),

    # --------------------------------------------------------- time ambiguity
    Case(
        id="no_timeframe",
        category="time",
        question="How is revenue doing?",
        must_mention_any=["90", "ninety", "assum", "default"],
        soft=True,
        note="Semantic layer says default to 90 days AND state that you did.",
    ),
    Case(
        id="last_quarter",
        category="time",
        question="How did we do last quarter?",
        must_mention_any=["quarter", "Q1", "Q2", "Q3", "Q4", "assum", "month"],
        soft=True,
        note="Calendar quarter or trailing 90 days? Either is fine if stated.",
    ),

    # ------------------------------------------------------------- ambiguity
    Case(
        id="how_many_customers",
        category="ambiguity",
        question="How many customers do we have?",
        must_mention_any=["registered", "signed up", "purchas", "bought", "assum", "defin"],
        soft=True,
        note="Registered users or paying users? Pick one and say which.",
    ),
    Case(
        id="best_product",
        category="ambiguity",
        question="What's our best product?",
        must_mention_any=["revenue", "units", "assum", "defin", "measur"],
        soft=True,
        note="Best by revenue, units, or margin? Should name the basis used.",
    ),

    # ----------------------------------------------------------- unanswerable
    Case(
        id="no_reviews",
        category="unanswerable",
        question="Which products have the best customer reviews?",
        forbid_number=True,
        note="No review data exists anywhere in the schema.",
    ),
    Case(
        id="no_margin",
        category="unanswerable",
        question="What is our profit margin per transaction?",
        forbid_number=True,
        note="No cost or fee column. Must not substitute revenue as a proxy.",
    ),
    Case(
        id="blocked_traits",
        category="unanswerable",
        question="Which users have the highest churn propensity score?",
        forbid_number=True,
        note="synthetic_user_traits is not readable by the agent role.",
    ),
    Case(
        id="no_ad_spend",
        category="unanswerable",
        question="How much did we spend on Google Ads last month?",
        forbid_number=True,
        note="acquisition_channel records paid_search but there is no spend data.",
    ),
]


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------

def _numeric_cells(result: db.QueryResult) -> list[float]:
    values = []
    for row in result.rows:
        for cell in row.values():
            if isinstance(cell, bool):
                continue
            if isinstance(cell, (int, float, Decimal)):
                values.append(float(cell))
            elif isinstance(cell, str):
                try:
                    values.append(float(cell.replace(",", "")))
                except ValueError:
                    pass
    return values


def _prose_numbers(text: str) -> list[float]:
    values = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return values


def _tables_in(sql: str) -> set[str]:
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return set()
    return {(t.name or "").lower() for t in tree.find_all(exp.Table)}


def _close(a: float, b: float) -> bool:
    return abs(a - b) / (abs(b) or 1.0) <= TOLERANCE


def _last_sql(state: dict) -> str | None:
    for message in reversed(state["messages"]):
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") == "run_sql":
                return call.get("args", {}).get("query")
    return None


@dataclass
class Outcome:
    hard_ok: bool = True
    soft_notes: list[str] = field(default_factory=list)
    detail: str = ""
    sql: str = ""
    answer: str = ""


def grade(case: Case, state: dict) -> Outcome:
    out = Outcome()
    out.answer = answer = state["messages"][-1].content or ""
    out.sql = sql = _last_sql(state) or ""

    # refusal cases
    if case.forbid_number:
        found = _prose_numbers(answer)
        if found:
            out.hard_ok = False
            out.detail = f"answered with a figure ({found[0]}) instead of declining"
        return out

    # value check
    if case.truth_sql:
        truth_result = db.run_query(case.truth_sql)
        expected = (
            float(list(truth_result.rows[0].values())[0])
            if truth_result.rows and list(truth_result.rows[0].values())[0] is not None
            else None
        )

        if expected is None:
            out.soft_notes.append("ground truth returned no rows; value check skipped")
        elif not sql:
            out.hard_ok = False
            out.detail = "agent never ran a query"
            return out
        else:
            try:
                agent_result = db.run_query(sql)
            except Exception as e:
                out.hard_ok = False
                out.detail = f"agent SQL does not execute: {e}"
                return out

            cells = _numeric_cells(agent_result)
            if not any(_close(c, expected) for c in cells):
                out.hard_ok = False
                out.detail = f"expected {expected}, query returned {cells[:6]}"
                return out

            if not any(_close(n, expected) for n in _prose_numbers(answer)):
                out.soft_notes.append(
                    f"SQL correct but the answer text never states {expected}"
                )

    # structural check
    if case.must_use_tables and sql:
        used = _tables_in(sql)
        missing = [t for t in case.must_use_tables if t.lower() not in used]
        if missing:
            message = f"did not query {', '.join(missing)} (used {', '.join(sorted(used)) or 'nothing'})"
            if case.soft:
                out.soft_notes.append(message)
            else:
                out.hard_ok = False
                out.detail = message
                return out

    # text checks — always soft
    lowered = answer.lower()
    for phrase in case.must_mention:
        if phrase.lower() not in lowered:
            out.soft_notes.append(f"answer never mentions {phrase!r}")
    if case.must_mention_any and not any(
        p.lower() in lowered for p in case.must_mention_any
    ):
        out.soft_notes.append(
            f"answer mentions none of {case.must_mention_any}"
        )

    return out


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run(verbose: bool = False, repeat: int = 1, category: str | None = None) -> int:
    agent = build_agent()
    selected = [c for c in CASES if not category or c.category == category]

    results: dict[str, list[Outcome]] = defaultdict(list)

    for attempt in range(repeat):
        if repeat > 1:
            print(f"\n=== run {attempt + 1} of {repeat} ===")

        current = ""
        for case in selected:
            if case.category != current:
                current = case.category
                print(f"\n  {current}")

            state = agent.invoke(
                {"messages": [{"role": "user", "content": case.question}]}
            )
            outcome = grade(case, state)
            results[case.id].append(outcome)

            if case.soft and not outcome.hard_ok:
                label = "SOFT"
            elif outcome.hard_ok and outcome.soft_notes:
                label = "PASS*"
            elif outcome.hard_ok:
                label = "PASS "
            else:
                label = "FAIL "

            print(f"  [{label}] {case.id}: {case.question[:64]}")

            if not outcome.hard_ok:
                print(f"          {outcome.detail}")
            for note in outcome.soft_notes:
                print(f"          soft: {note}")

            if verbose or (not outcome.hard_ok and not case.soft):
                if case.note:
                    print(f"          note: {case.note}")
                if outcome.sql:
                    print(f"          sql:  {' '.join(outcome.sql.split())[:200]}")
                print(f"          said: {outcome.answer.strip()[:200]}\n")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 68)
    hard_cases = [c for c in selected if not c.soft]
    failed = [c for c in hard_cases if not all(o.hard_ok for o in results[c.id])]
    flaky = [
        c for c in selected
        if len({o.hard_ok for o in results[c.id]}) > 1
    ]
    soft_issues = [
        c for c in selected
        if any(o.soft_notes or not o.hard_ok for o in results[c.id]) and c.soft
    ]

    print(f"hard: {len(hard_cases) - len(failed)}/{len(hard_cases)} passed")
    if failed:
        print("  failed: " + ", ".join(c.id for c in failed))
    if soft_issues:
        print(f"soft:  {len(soft_issues)} behavioural note(s): "
              + ", ".join(c.id for c in soft_issues))
    if flaky:
        print(
            f"\nFLAKY across runs: {', '.join(c.id for c in flaky)}\n"
            "  A case that changes verdict between identical runs was never "
            "really passing. Fix these before trusting the suite."
        )
    elif repeat > 1:
        print("no flakiness detected across runs")

    print(
        "\nWhen a hard case fails: read the sql line, diff it against the case's "
        "truth_sql,\nand fix the RULE in semantic_layer.yaml — not this file. "
        "Then check the\nmatching paraphrase case still passes."
    )
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the suite N times to detect flaky cases")
    parser.add_argument("--category", help="run one category only")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    args = parser.parse_args()

    if args.list:
        current = ""
        for case in CASES:
            if case.category != current:
                current = case.category
                print(f"\n{current}")
            kind = "soft" if case.soft else ("refusal" if case.forbid_number else "value")
            print(f"  {case.id:28} [{kind}] {case.question}")
        print(f"\n{len(CASES)} cases, "
              f"{len({c.category for c in CASES})} categories")
        return 0

    return run(args.verbose, args.repeat, args.category)


if __name__ == "__main__":
    raise SystemExit(main())
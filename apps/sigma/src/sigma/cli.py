"""Command line interface for SIGMA.

The CLI is the human-in-the-loop surface for terminal users: when the graph
interrupts, the operator answers here and the run continues from the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import Any

from sigma.runtime import get_runner
from sigma.size import audit as size_audit
from sigma.state import HumanDecision, RunStatus

QUESTION_EXAMPLE = "Bagaimana risiko dan alokasi modal portofolio ventura kita kuartal ini?"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _print_result(result: Any) -> None:
    state = result.state or {}
    brief = state.get("brief")
    print(f"\nsession : {result.session_id}")
    print(f"status  : {result.status.value}")
    if result.error:
        print(f"error   : {result.error}")
    evaluation = state.get("evaluation") or []
    if evaluation:
        latest = evaluation[-1]
        print(f"score   : {latest.score:.3f} ({latest.verdict}) "
              f"after {len(evaluation)} evaluation(s)")
    if brief:
        print("\n" + textwrap.fill(brief.headline, 88))
        print("-" * 88)
        for line in brief.summary.split(". "):
            print(textwrap.fill(line.strip() + ("" if line.endswith(".") else "."), 88))
        print("-" * 88)
        for index, recommendation in enumerate(brief.recommendations, start=1):
            print(textwrap.fill(f"{index}. {recommendation}", 88, subsequent_indent="   "))
        if brief.caveats:
            print("\ncatatan:")
            for caveat in brief.caveats:
                print(textwrap.fill(f"  ! {caveat}", 88, subsequent_indent="    "))


def _ask_human(result: Any) -> HumanDecision | None:
    """Terminal HITL prompt."""
    for interrupt in result.interrupts:
        for item in interrupt.get("interrupts", []):
            value = item.get("value", {})
            print("\n" + "=" * 88)
            print("PERSETUJUAN MANUSIA DIPERLUKAN")
            print("=" * 88)
            print(f"skor      : {value.get('score')}")
            print(f"ringkasan : {value.get('summary')}")
            print("rekomendasi:")
            for line in value.get("recommendations", []):
                print(textwrap.fill(f"  - {line}", 86, subsequent_indent="    "))
            print(f"gap       : {len(value.get('open_gaps', []))}")
            print("=" * 88)
    raw = input("keputusan [approve/revise/reject] (default approve): ").strip().lower()
    decision = raw or "approve"
    if decision not in ("approve", "revise", "reject"):
        print(f"keputusan tidak dikenal: {decision!r} -> diperlakukan sebagai reject")
        decision = "reject"
    note = ""
    if decision in ("revise", "reject"):
        note = input("catatan untuk agen: ").strip()
    return HumanDecision(decision=decision, note=note, reviewer="cli")


def cmd_run(args: argparse.Namespace) -> int:
    runner = get_runner(hitl=args.hitl)
    result = runner.start(question=args.question, seed=args.seed,
                          portfolio_id=args.portfolio_id,
                          acceptance_threshold=args.threshold,
                          max_revisions=args.max_revisions)

    if args.stream:
        pass  # node-by-node output is emitted below on the next run
    for event in (result.state.get("audit_trail") or []):
        print(f"[{event.seq:>3}] {event.node:<12} {event.kind:<18} {event.message[:70]}")

    while result.status is RunStatus.AWAITING_HUMAN and not args.no_hitl:
        decision = _ask_human(result)
        if decision is None:
            break
        result = runner.resume(result.session_id, decision)
        for event in (result.state.get("audit_trail") or [])[-6:]:
            print(f"[{event.seq:>3}] {event.node:<12} {event.kind:<18} {event.message[:70]}")

    _print_result(result)
    return 0 if result.status in (RunStatus.COMPLETED, RunStatus.REJECTED) else 1


def cmd_sessions(args: argparse.Namespace) -> int:
    from sigma.api.serialization import to_jsonable
    from sigma.persistence.sessions import session_summaries

    summaries = session_summaries(get_runner(hitl=False), limit=args.limit)
    rows = [{"session": s["session_id"][:12], "status": s["status"],
             "score": s["score"], "rev": s["revision"], "evidence": s["evidence"],
             "question": (s["question"] or "")[:48]} for s in summaries]
    _print_json(rows if not args.full else to_jsonable(summaries))
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    from sigma.api.serialization import timeline_view

    runner = get_runner(hitl=False)
    _print_json(timeline_view(runner.history(args.session_id)))
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    from sigma.api.serialization import session_view

    runner = get_runner(hitl=False)
    snapshot = runner.state(args.session_id)
    if snapshot is None:
        print(f"unknown session {args.session_id}", file=sys.stderr)
        return 1
    _print_json(session_view(dict(snapshot.values or {})))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    runner = get_runner(hitl=True)
    result = runner.resume(args.session_id,
                           HumanDecision(decision=args.decision, note=args.note,
                                         reviewer="cli"))
    _print_result(result)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from sigma.api.app import create_app

    app = create_app(hitl=not args.no_hitl)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_audit_size(args: argparse.Namespace) -> int:
    report = size_audit.measure()
    print(size_audit.render(report))
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigma",
        description="SIGMA — platform analitik multi-agen berbasis LangGraph.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="jalankan sesi analitik")
    run.add_argument("--question", default=QUESTION_EXAMPLE)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--portfolio-id", default="sigma-core")
    run.add_argument("--threshold", type=float, default=None)
    run.add_argument("--max-revisions", type=int, default=None)
    run.add_argument("--no-hitl", action="store_true", help="lewati gate persetujuan manusia")
    run.add_argument("--hitl", action="store_true", default=True)
    run.add_argument("--stream", action="store_true", help="(kompatibilitas) selalu mencatat audit")
    run.set_defaults(func=cmd_run)

    sessions = sub.add_parser("sessions", help="daftar sesi tersimpan")
    sessions.add_argument("--limit", type=int, default=20)
    sessions.add_argument("--full", action="store_true")
    sessions.set_defaults(func=cmd_sessions)

    timeline = sub.add_parser("timeline", help="riwayat checkpoint (time-travel)")
    timeline.add_argument("session_id")
    timeline.set_defaults(func=cmd_timeline)

    state = sub.add_parser("state", help="tampilkan state sesi")
    state.add_argument("session_id")
    state.set_defaults(func=cmd_state)

    resume = sub.add_parser("resume", help="jawab interrupt HITL")
    resume.add_argument("session_id")
    resume.add_argument("--decision", choices=("approve", "revise", "reject"),
                        default="approve")
    resume.add_argument("--note", default="")
    resume.set_defaults(func=cmd_resume)

    serve = sub.add_parser("serve", help="jalankan antarmuka web")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--no-hitl", action="store_true")
    serve.set_defaults(func=cmd_serve)

    audit = sub.add_parser("audit-size", help="audit ukuran artefak terhadap budget 125 MB")
    audit.set_defaults(func=cmd_audit_size)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

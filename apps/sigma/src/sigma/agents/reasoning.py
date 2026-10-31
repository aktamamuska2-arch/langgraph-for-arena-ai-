"""Deterministic reasoning policies used by the agents.

Every function here is a pure function of the graph state (+ a seed), which is
what makes NFR-3 (identical seed -> identical score) achievable and the whole
test suite hermetic.

In FASE 2 these same policies are exposed through the LLM provider abstraction
(`sigma.llm.DeterministicProvider`) so that a networked model can be swapped in
without touching a single node.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from sigma.config import SETTINGS
from sigma.state import (
    DraftBrief,
    Evaluation,
    Evidence,
    Finding,
    Gap,
    PlanState,
    RequestPayload,
    RunStatus,
    ScenarioContext,
    SubQuestion,
    ToolCall,
    active_findings,
)

# --------------------------------------------------------------------------- #
# Intent detection — drives which sub-questions the planner asks
# --------------------------------------------------------------------------- #

_SECTOR_HINTS = ("fintech", "logistics", "healthtech", "agritech", "edutech", "climate")
_INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "risk": ("risiko", "risk", "downside", "kerugian", "var", "volatil"),
    "growth": ("pertumbuhan", "growth", "return", "imbal", "multiple", "untung"),
    "allocation": ("alokasi", "allocate", "modal", "capital", "invest", "deploy"),
    "diversification": ("diversifikasi", "konsentrasi", "concentration", "diversif", "hhi"),
    "benchmark": ("benchmark", "banding", "kuartil", "sektor", "sector", "kompetitor"),
}


def detect_intents(question: str) -> list[str]:
    """Return the ordered list of analytical intents found in the question."""
    low = question.lower()
    found = [name for name, hints in _INTENT_HINTS.items()
             if any(h in low for h in hints)]
    return found or ["risk", "growth", "allocation"]


def detect_sectors(question: str, scenario: ScenarioContext) -> list[str]:
    low = question.lower()
    found = [s for s in _SECTOR_HINTS if s in low]
    return found or list(scenario.sectors)[:2]


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #

_TEMPLATES: dict[str, tuple[str, str, str]] = {
    # intent: (question template, tool hint, rationale)
    "allocation": ("Bagaimana komposisi dan sebaran modal portofolio saat ini?",
                   "query_portfolio", "Menetapkan basis eksposur sebelum menilai alokasi."),
    "diversification": ("Seberapa terkonsentrasi eksposur portofolio (HHI) dan apa dampaknya?",
                        "compute_risk_metrics", "Konsentrasi mengubah profil risiko secara non-linear."),
    "risk": ("Berapa skor risiko dan volatilitas tertimbang pada eksposur utama?",
             "compute_risk_metrics", "Mengukur risiko pada posisi terbesar tempat kerugian nyata terjadi."),
    "growth": ("Berapa nilai akhir yang diharapkan (expected multiple) pada horizon analisis?",
               "scenario_simulate", "Estimasi titik-tengah hasil untuk menilai kelayakan imbal hasil."),
    "benchmark": ("Bagaimana metrik portofolio dibandingkan kuartil sektornya?",
                  "benchmark_sector", "Tanpa basis sektor, angka absolut tidak dapat dinilai."),
}


def plan_questions(
    request: RequestPayload,
    scenario: ScenarioContext,
    *,
    revision: int = 0,
    directive: str | None = None,
    existing: Iterable[SubQuestion] = (),
    max_questions: int | None = None,
) -> PlanState:
    """Decompose the request into sub-questions.

    On a revision the planner *keeps* what is already answered and adds targeted
    questions for the directive it was given (critic gaps or a human note) — the
    mechanism that makes the self-correction cycle converge instead of restarting.
    """
    cap = max_questions or SETTINGS.max_sub_questions
    intents = detect_intents(request.question)
    sectors = detect_sectors(request.question, scenario)

    kept = [sq for sq in existing if sq.status != "dropped"]
    questions: list[SubQuestion] = list(kept)

    def add(intent: str, question: str, tool: str, rationale: str, priority: float) -> None:
        if len(questions) >= cap:
            return
        if any(q.question == question for q in questions):
            return
        questions.append(SubQuestion(id=f"q{len(questions) + 1}", question=question,
                                     tool_hint=tool, rationale=rationale, priority=priority))

    # Base decomposition from detected intents.
    core = ["allocation", "risk", "growth", "diversification", "benchmark"]
    ordered = [i for i in core if i in intents] + [i for i in core if i not in intents]
    for rank, intent in enumerate(ordered):
        template, tool, rationale = _TEMPLATES[intent]
        add(intent, template, tool, rationale, priority=round(1.0 - rank * 0.1, 2))

    # Sector follow-up when the user named a sector explicitly.
    for sector in sectors[:1]:
        add("benchmark", f"Apakah metrik sektor {sector} berada di atas kuartil tengahnya?",
            "benchmark_sector", f"Sektor {sector} disebut eksplisit dalam permintaan.",
            priority=0.6)

    # Directive-driven follow-ups (this is where the cycle adds information).
    if directive:
        low = directive.lower()
        if "evidence" in low or "bukti" in low or "missing_evidence" in low:
            add("risk", "Ventura mana yang belum memiliki metrik risiko terukur?",
                "compute_risk_metrics", "Kritik/Evaluator: evidensi belum lengkap.", priority=0.95)
        if "structural" in low or "struktur" in low or "fokus" in low:
            add("growth", "Apakah hipotesis pertumbuhan masih berlaku setelah revisi?",
                "scenario_simulate", "Kritik/Evaluator: kerangka pertanyaannya perlu diganti.",
                priority=0.95)
        if "numerical" in low or "numeric" in low or "angka" in low:
            add("growth", "Perkuat estimasi dengan simulasi ber paths lebih tinggi.",
                "scenario_simulate", "Kritik/Evaluator: angka belum terdukung cukup bukti.",
                priority=0.9)
        if "interpretation" in low or "interpretasi" in low or "rekomendasi" in low:
            add("benchmark", "Apa implikasi aksi dari perbandingan sektor?",
                "benchmark_sector", "Kritik/Evaluator: rekomendasi masih generik.", priority=0.9)
        add("allocation", f"Tindak lanjut instruksi: {directive[:80]}",
            "query_portfolio", "Instruksi eksplisit dari evaluator/manusia.", priority=0.99)

    # Keep only the highest priority questions within the cap.
    questions.sort(key=lambda q: (-q.priority, q.id))
    questions = questions[:cap]

    hypothesis = (
        f"Portofolio {scenario.portfolio_id} dengan {scenario.venture_count} ventura: "
        f"risiko terkonsentrasi pada eksposur terbesar, sehingga keputusan alokasi harus "
        f"didorong oleh risiko ekor (tail risk), bukan imbal hasil rata-rata."
    )
    return PlanState(revision=revision, hypothesis=hypothesis,
                     sub_questions=questions, directive=directive)


# --------------------------------------------------------------------------- #
# Researcher — tool selection
# --------------------------------------------------------------------------- #


def _known_venture_ids(evidence: list[Evidence], limit: int = 6) -> list[str]:
    """Best-known venture ids, taken from evidence (never from a hidden shortcut)."""
    ids: list[str] = []
    for item in reversed(evidence):
        for venture in item.payload.get("ventures", []) or []:
            vid = venture.get("id")
            if vid and vid not in ids:
                ids.append(vid)
    if ids:
        return ids[:limit]
    # Fall back to ids echoed by risk/scenario evidence (best effort, still evidence-based).
    return []


def _evidence_signature(tool: str, args: dict[str, Any]) -> str:
    norm = {k: (tuple(sorted(v)) if isinstance(v, list) else v) for k, v in sorted(args.items())}
    return f"{tool}:{norm!r}"


def satisfied_question_ids(state: dict[str, Any]) -> set[str]:
    """Sub-questions already answered — directly, or by an identical executed call.

    A question is satisfied when some evidence cites it **or** when an identical
    `(tool, args)` call has already produced evidence for another question. That
    second rule is the deduplication contract: without it, a revision would keep
    re-asking a question whose answer is already in the evidence set, coverage
    would never reach 1.0 and the cycle would spin forever.
    """
    plan: PlanState | None = state.get("plan")
    if plan is None:
        return set()
    evidence: list[Evidence] = list(state.get("evidence") or [])
    executed = {(c.tool, _evidence_signature(c.tool, dict(c.args)))
                for c in (state.get("tool_calls") or []) if c.ok}
    answered: set[str] = {e.sub_question_id for e in evidence if e.sub_question_id}
    ids = _known_venture_ids(evidence)
    for sq in plan.sub_questions:
        if sq.id in answered or sq.status == "dropped":
            continue
        signature = _evidence_signature(
            sq.tool_hint, _args_for(sq.tool_hint, ids, state, plan.revision))
        if (sq.tool_hint, signature) in executed:
            answered.add(sq.id)
    return answered


def propose_tool_calls(state: dict[str, Any], *, budget_left: int) -> list[ToolCall]:
    """Pick the next tool calls for unanswered sub-questions.

    Returns an empty list once there is nothing left to gather — that is the
    signal the researcher uses to leave its self-loop.
    """
    plan: PlanState | None = state.get("plan")
    if plan is None:
        return []
    tool_calls = list(state.get("tool_calls") or [])
    done_signatures = {
        _evidence_signature(c.tool, dict(c.args)) for c in tool_calls if c.ok
    }
    answered = satisfied_question_ids(state)
    ids = _known_venture_ids(list(state.get("evidence") or []))
    depth = int(state.get("research_pass") or 0)

    def _add(sq: SubQuestion, at_depth: int) -> bool:
        if len(calls) >= budget_left:
            return False
        args = _args_for(sq.tool_hint, ids, state, at_depth)
        signature = _evidence_signature(sq.tool_hint, args)
        if signature in done_signatures:
            return False  # identical call already executed -> nothing new
        calls.append(ToolCall(tool=sq.tool_hint, args=args, sub_question_id=sq.id))
        done_signatures.add(signature)
        return True

    calls: list[ToolCall] = []
    for sq in sorted(plan.sub_questions, key=lambda q: -q.priority):
        if sq.status == "dropped" or sq.id in answered:
            continue
        _add(sq, depth)

    if not calls and needs_more_evidence(state):
        # Every question is answered but the critic says the evidence is too thin:
        # re-query the highest-priority questions one level deeper (wider venture
        # set, longer horizon, more Monte-Carlo paths). This is the mechanism that
        # makes the macro-cycle gain information instead of idling.
        for sq in sorted(plan.sub_questions, key=lambda q: -q.priority):
            if sq.status == "dropped":
                continue
            if _add(sq, depth + 1):
                break
    return calls


def needs_more_evidence(state: dict[str, Any]) -> bool:
    """True when the latest critique asked for more/numerically stronger evidence."""
    evaluations = list(state.get("evaluation") or [])
    if not evaluations:
        return False
    latest = evaluations[-1]
    return latest.verdict == "revise" and any(
        g.kind in ("missing_evidence", "numerical") for g in latest.gaps)


def _args_for(tool: str, ids: list[str], state: dict[str, Any],
              revision: int = 0) -> dict[str, Any]:
    """Build tool arguments for a sub-question.

    `revision` widens the query (more ventures, longer horizon, more Monte-Carlo
    paths, next sector). That is what makes each self-correction cycle produce
    *new* evidence instead of re-running the previous call: the cycle converges
    because the evidence set genuinely grows (PRD NFR-4).
    """
    scenario: ScenarioContext | None = state.get("scenario")
    horizon = scenario.horizon_quarters if scenario else 8
    sectors = list(scenario.sectors) if scenario else []
    depth = max(0, min(revision, 4))
    fallback = ["V01", "V02", "V03"]

    if tool == "query_portfolio":
        args: dict[str, Any] = {"limit": 12}
        if depth and sectors:
            args["sector"] = sectors[depth % len(sectors)]
        return args
    if tool == "compute_risk_metrics":
        return {"venture_ids": (ids[: 6 + depth * 3] if ids else fallback),
                "horizon_quarters": horizon + depth * 4}
    if tool == "scenario_simulate":
        paths = min(6_000, SETTINGS.monte_carlo_paths * (1 + depth))
        return {"venture_ids": (ids[:6] if ids else fallback),
                "paths": paths, "horizon_quarters": horizon}
    if tool == "benchmark_sector":
        sector = detect_sectors(
            (state.get("request").question if state.get("request") else ""), scenario
        ) if scenario else []
        if depth and sectors:
            return {"sector": sectors[depth % len(sectors)]}
        return {"sector": sector[0] if sector else (sectors[0] if sectors else "fintech")}
    return {}


# --------------------------------------------------------------------------- #
# Analyst — evidence -> findings -> draft
# --------------------------------------------------------------------------- #


def build_findings(state: dict[str, Any]) -> list[Finding]:
    """Turn evidence into quantified findings; later findings supersede earlier ones."""
    evidence: list[Evidence] = list(state.get("evidence") or [])
    previous = {(f.sub_question_id, f.unit): f for f in active_findings(state)}
    # Evidence already cited by a live finding must not be re-derived on every
    # cycle, otherwise the finding list grows with the number of supersteps.
    cited = {eid for f in active_findings(state) for eid in f.evidence_ids}
    findings: list[Finding] = []

    def add(sq_id: str | None, claim: str, metric: float | None, unit: str | None,
            evidence_ids: list[str], confidence: float) -> None:
        prior = previous.get((sq_id, unit))
        findings.append(Finding(
            sub_question_id=sq_id, claim=claim, metric=metric, unit=unit,
            evidence_ids=evidence_ids,
            confidence=round(max(0.05, min(0.99, confidence)), 3),
            supersedes=prior.finding_id if prior else None))

    for item in evidence:
        if item.evidence_id in cited:
            continue
        payload = item.payload
        if item.tool == "compute_risk_metrics" and "risk_score" in payload:
            add(item.sub_question_id,
                f"Skor risiko tertimbang eksposur utama {payload['risk_score']:.2f} "
                f"(volatilitas {payload['weighted_volatility']:.2f}).",
                float(payload["risk_score"]), "risk_score", [item.evidence_id],
                confidence=0.55 + 0.25 * min(1.0, len(item.args.get("venture_ids", [])) / 4))
            add(item.sub_question_id,
                f"Konsentrasi portofolio HHI {payload['concentration_hhi']:.3f} pada "
                f"{payload['venture_count']} ventura.",
                float(payload["concentration_hhi"]), "hhi", [item.evidence_id], confidence=0.8)
            add(item.sub_question_id,
                f"Expected multiple {payload['expected_multiple']:.2f}x dari modal "
                f"{payload['invested']} juta USD.",
                float(payload["expected_multiple"]), "multiple", [item.evidence_id], confidence=0.75)
        elif item.tool == "scenario_simulate" and "p50_multiple" in payload:
            add(item.sub_question_id,
                f"Simulasi {payload['paths']} jalur: P50 {payload['p50_multiple']:.2f}x, "
                f"P10 {payload['p10_multiple']:.2f}x, VaR95 {payload['var95']:.2f}, "
                f"P(rugi) {payload['prob_loss']:.0%}.",
                float(payload["p50_multiple"]), "p50_multiple", [item.evidence_id],
                confidence=0.6 + 0.2 * min(1.0, payload["paths"] / 4000))
        elif item.tool == "query_portfolio" and payload.get("ventures"):
            add(item.sub_question_id,
                f"Portofolio memuat {payload['count']} ventura dengan total modal "
                f"{payload['total_capital']} juta USD (median {payload['median_capital']}).",
                float(payload["total_capital"]), "usd_millions", [item.evidence_id], confidence=0.9)
        elif item.tool == "benchmark_sector" and "growth_quartiles" in payload:
            add(item.sub_question_id,
                f"Kuartil pertumbuhan sektor {payload['sector']}: p25 "
                f"{payload['growth_quartiles']['p25']}, p50 {payload['growth_quartiles']['p50']}, "
                f"p75 {payload['growth_quartiles']['p75']} (n={payload['sample_size']}).",
                float(payload["growth_quartiles"]["p50"]), "growth_p50", [item.evidence_id],
                confidence=0.7)
    return findings


def build_draft(state: dict[str, Any], findings: list[Finding]) -> DraftBrief:
    """Compose the draft brief the critic will score."""
    plan: PlanState | None = state.get("plan")
    revision = plan.revision if plan else 0
    by_unit = {f.unit: f for f in findings if f.metric is not None}
    risk = by_unit.get("risk_score")
    hhi = by_unit.get("hhi")
    p50 = by_unit.get("p50_multiple")
    multiple = by_unit.get("multiple")
    total = by_unit.get("usd_millions")

    recommendations: list[str] = []
    caveats: list[str] = []
    if risk is not None:
        if risk.metric >= 0.6:
            recommendations.append(
                f"Kurangi eksposur pada ventura berisiko tinggi (skor risiko {risk.metric:.2f}) "
                f"atau pasang proteksi lewat tranche bertahap.")
        else:
            recommendations.append(
                f"Profil risiko moderat (skor {risk.metric:.2f}); alokasi dapat dipertahankan "
                f"dengan monitoring kuartalan.")
    if hhi is not None and hhi.metric >= 0.22:
        recommendations.append(
            f"Diversifikasi: HHI {hhi.metric:.3f} menunjukkan eksposur terkonsentrasi; "
            f"tambah ventura di luar 3 posisi terbesar.")
        caveats.append("Konsentrasi tinggi menurunkan reliabilitas estimasi risiko portofolio.")
    if p50 is not None and p50.metric < 1.0:
        recommendations.append(
            f"Tahan pencairan lanjutan: median hasil simulasi {p50.metric:.2f}x berada di bawah modal.")
        caveats.append("Median hasil di bawah 1.0x pada horizon analisis.")
    if multiple is not None and multiple.metric >= 1.15:
        recommendations.append(
            f"Pertahankan alokasi pada posisi dengan expected multiple {multiple.metric:.2f}x.")
    if not recommendations:
        recommendations.append("Kumpulkan ulang metrik inti sebelum mengambil keputusan alokasi.")
        caveats.append("Rekomendasi bersifat provisional karena bukti belum lengkap.")

    headline = (
        f"Rekomendasi alokasi portofolio — risiko {'tinggi' if (risk and risk.metric >= 0.6) else 'moderat'}"
        f", {len(recommendations)} tindakan terprioritas"
    )
    summary = (
        f"Portofolio bernilai {total.metric if total else 0:.1f} juta USD dievaluasi pada horizon "
        f"{state.get('scenario').horizon_quarters if state.get('scenario') else 8} kuartal. "
        f"{len(findings)} temuan kuantitatif dihasilkan dari "
        f"{len(state.get('evidence') or [])} bukti terverifikasi. "
        f"{'Risiko ekor menjadi pertimbangan utama.' if (p50 and p50.metric < 1.0) else 'Imbal hasil sejalan dengan target.'}"
    )
    return DraftBrief(revision=revision, headline=headline, summary=summary,
                      recommendations=recommendations[:4],
                      finding_ids=[f.finding_id for f in findings], caveats=caveats[:3])


# --------------------------------------------------------------------------- #
# Critic — rubric scoring
# --------------------------------------------------------------------------- #

_RUBRIC_WEIGHTS = {
    "coverage": 0.25,
    "evidence_density": 0.25,
    "numeric_grounding": 0.20,
    "actionability": 0.15,
    "calibration": 0.15,
}


def critique(state: dict[str, Any], *, threshold: float | None = None) -> Evaluation:
    """Score the current draft against the rubric and enumerate concrete gaps.

    The score is computed from real artefacts (sub-questions, evidence, findings,
    recommendations) — not from a constant — so the critique reacts to the run and
    the self-correction loop has something genuine to fix.
    """
    limit = SETTINGS.acceptance_threshold if threshold is None else threshold
    plan: PlanState | None = state.get("plan")
    draft: DraftBrief | None = state.get("draft")
    evidence: list[Evidence] = list(state.get("evidence") or [])
    findings: list[Finding] = active_findings(state)
    questions = [q for q in ((plan.sub_questions if plan else [])) if q.status != "dropped"]
    answered = satisfied_question_ids(state)

    coverage = (sum(1 for q in questions if q.id in answered) / len(questions)) if questions else 0.0
    evidence_density = min(1.0, len(evidence) / max(2.0, 2.0 * len(questions))) if questions else 0.0
    numeric = [f for f in findings if f.metric is not None]
    grounded = [f for f in numeric if f.evidence_ids]
    numeric_grounding = (len(grounded) / len(numeric)) if numeric else 0.0
    recs = draft.recommendations if draft else []
    actionability = min(1.0, len(recs) / 3.0) * (1.0 if any(len(r) > 40 for r in recs) else 0.7)
    # Calibration: confidence must be supported by the amount of evidence behind it.
    if findings:
        calibration = sum(
            min(1.0, len(f.evidence_ids) / 2.0) * (1.0 - abs(f.confidence - 0.7))
            for f in findings) / len(findings)
        calibration = max(0.0, min(1.0, calibration + 0.35))
    else:
        calibration = 0.0

    scores = _scores(coverage, evidence_density, numeric_grounding, actionability, calibration)
    score = _weighted(scores)
    gaps = _gaps(state, coverage, evidence_density, numeric_grounding, actionability, questions, answered)

    return Evaluation(
        revision=plan.revision if plan else 0,
        score=score,
        scores=scores,
        gaps=gaps,
        verdict="accept" if score >= limit else "revise",
        rationale=_rationale(score, limit, gaps),
    )


def _scores(coverage: float, density: float, grounding: float,
            actionability: float, calibration: float) -> Any:
    from sigma.state import RubricScores

    return RubricScores(
        coverage=round(coverage, 4),
        evidence_density=round(density, 4),
        numeric_grounding=round(grounding, 4),
        actionability=round(actionability, 4),
        calibration=round(calibration, 4),
    )


def _weighted(scores: Any) -> float:
    total = sum(_RUBRIC_WEIGHTS[name] * getattr(scores, name) for name in _RUBRIC_WEIGHTS)
    return round(total, 4)


def _gaps(state: dict[str, Any], coverage: float, density: float, grounding: float,
          actionability: float, questions: list[SubQuestion],
          answered: set[str | None]) -> list[Gap]:
    gaps: list[Gap] = []
    if coverage < 0.85:
        for q in questions:
            if q.id not in answered:
                gaps.append(Gap(kind="missing_evidence",
                                detail=f"Sub-pertanyaan belum terjawab: {q.question}",
                                sub_question_id=q.id, weight=1.0))
    if density < 0.7:
        gaps.append(Gap(kind="missing_evidence",
                        detail=f"Kepadatan evidensi {density:.2f} < 0.70; tambah pemanggilan tool.",
                        weight=1.2))
    if grounding < 0.85:
        gaps.append(Gap(kind="numerical",
                        detail=f"Hanya {grounding:.0%} temuan numerik yang memiliki bukti.",
                        weight=1.1))
    if actionability < 0.7:
        gaps.append(Gap(kind="interpretation",
                        detail="Rekomendasi terlalu sedikit/generik; perlu interpretasi aksi.",
                        weight=1.0))
    if coverage < 0.5:
        gaps.append(Gap(kind="structural",
                        detail="Kerangka pertanyaan tidak menutup permintaan; susun ulang dekomposisi.",
                        weight=1.4))
    if not gaps:
        gaps.append(Gap(kind="structural", detail="Tidak ada gap material.", weight=0.0))
    return gaps


def _rationale(score: float, limit: float, gaps: list[Gap]) -> str:
    material = [g for g in gaps if g.weight > 0]
    if score >= limit:
        return f"Skor {score:.3f} >= ambang {limit:.2f}: brief layak diajukan ke manusia."
    detail = "; ".join(f"[{g.kind}] {g.detail}" for g in material[:3])
    return f"Skor {score:.3f} < ambang {limit:.2f}. Gap: {detail or 'n/a'}"


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def build_final_brief(state: dict[str, Any], status: RunStatus) -> Any:
    from sigma.state import FinalBrief

    evaluation = (state.get("evaluation") or [])
    draft: DraftBrief | None = state.get("draft")
    findings = active_findings(state)
    evidence = list(state.get("evidence") or [])
    plan: PlanState | None = state.get("plan")
    latest = evaluation[-1] if evaluation else None
    return FinalBrief(
        session_id=state.get("session_id", ""),
        status=status,
        headline=(draft.headline if draft else "Brief tidak tersedia"),
        summary=(draft.summary if draft else ""),
        recommendations=(draft.recommendations if draft else []),
        findings=findings,
        score=(latest.score if latest else 0.0),
        revisions=(plan.revision if plan else 0),
        caveats=(draft.caveats if draft else []),
        evidence_count=len(evidence),
        provenance=[e.provenance for e in evidence][:50],
    )

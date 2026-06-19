from __future__ import annotations

from .candidates import generate_candidates
from .graph import DocumentGraph
from .reranking import CandidateReranker
from .types import FieldRequest, GroundedField, GroundingCandidate


class GroundingEngine:
    def __init__(self, *, reranker: CandidateReranker | None = None, top_k: int = 20) -> None:
        self.reranker = reranker or CandidateReranker()
        self.top_k = top_k

    def ground(self, graph: DocumentGraph, requests: list[FieldRequest]) -> list[GroundedField]:
        return [self._ground_one(graph, request) for request in requests]

    def _ground_one(self, graph: DocumentGraph, request: FieldRequest) -> GroundedField:
        candidates = generate_candidates(graph, request, top_k=self.top_k)
        candidates = self.reranker.score(request, candidates, graph)
        candidates = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
        if not candidates:
            return GroundedField(
                field_id=request.field_id,
                key=request.key,
                value=request.value,
                matched_text=None,
                page=None,
                bbox=None,
                confidence=0.0,
                status="not_found",
                source="grounding_engine",
                row_index=request.row_index,
                evidence={"reason": "No structurally valid candidates found"},
            )
        best = candidates[0]
        status = _status(best, candidates)
        return GroundedField(
            field_id=request.field_id,
            key=request.key,
            value=request.value,
            matched_text=best.matched_text,
            page=best.page,
            bbox=best.value_box,
            confidence=best.confidence,
            status=status,
            source=best.relationship.value,
            row_index=best.row_index,
            col_index=best.col_index,
            table_id=best.table_id,
            evidence={
                **(best.evidence or {}),
                "candidate_id": best.candidate_id,
                "alternatives": [_candidate_summary(candidate) for candidate in candidates[1:4]],
            },
        )


def _status(best: GroundingCandidate, candidates: list[GroundingCandidate]) -> str:
    if best.confidence < 0.6:
        return "low_confidence"
    if len(candidates) > 1 and best.confidence - candidates[1].confidence < 0.05:
        return "ambiguous"
    return "matched"


def _candidate_summary(candidate: GroundingCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "matched_text": candidate.matched_text,
        "confidence": candidate.confidence,
        "page": candidate.page,
        "row_index": candidate.row_index,
        "bbox": candidate.value_box.to_list(),
        "relationship": candidate.relationship.value,
    }

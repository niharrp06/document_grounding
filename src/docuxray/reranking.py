from __future__ import annotations

from collections.abc import Sequence

from .graph import DocumentGraph
from .types import FieldRequest, GroundingCandidate


class CandidateReranker:
    def score(
        self,
        request: FieldRequest,
        candidates: Sequence[GroundingCandidate],
        graph: DocumentGraph,
    ) -> list[GroundingCandidate]:
        return list(candidates)


class LayoutLMv3Reranker(CandidateReranker):
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir

    def score(
        self,
        request: FieldRequest,
        candidates: Sequence[GroundingCandidate],
        graph: DocumentGraph,
    ) -> list[GroundingCandidate]:
        raise NotImplementedError(
            "LayoutLMv3 pairwise reranking is intentionally isolated behind this interface. "
            "Train a candidate reranker, then implement this method without changing the grounding engine."
        )

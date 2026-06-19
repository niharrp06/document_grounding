from __future__ import annotations

from difflib import SequenceMatcher

from .graph import DocumentGraph
from .matching import box_for_indices, find_value_span
from .normalization import normalize_text, normalize_value
from .types import (
    CandidateRelationship,
    FieldRequest,
    GroundingCandidate,
)


def generate_candidates(
    graph: DocumentGraph,
    request: FieldRequest,
    *,
    top_k: int = 20,
) -> list[GroundingCandidate]:
    candidates = []
    candidates.extend(_table_candidates(graph, request))
    candidates.extend(_line_candidates(graph, request))
    candidates.extend(_value_only_candidates(graph, request))
    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    return candidates[:top_k]


def _table_candidates(graph: DocumentGraph, request: FieldRequest) -> list[GroundingCandidate]:
    candidates: list[GroundingCandidate] = []
    key_norm = normalize_text(request.key)
    target_value = normalize_value(request.value, request.value_type)
    for cell in graph.cells:
        if cell.is_header:
            continue
        if request.row_index is not None and cell.row_index != request.row_index:
            continue
        value_score = _ratio(normalize_value(cell.text, request.value_type), target_value)
        if value_score < 0.82:
            continue
        header_score = _ratio(normalize_text(cell.header_text or ""), key_norm)
        if header_score < 0.6:
            continue
        row_score = 1.0 if request.row_index is None or request.row_index == cell.row_index else 0.0
        confidence = 0.55 * value_score + 0.30 * header_score + 0.15 * row_score
        candidates.append(
            GroundingCandidate(
                candidate_id=f"{request.field_id}:table:{cell.cell_id}",
                field_id=request.field_id,
                key=request.key,
                value=request.value,
                matched_text=cell.text,
                page=cell.page,
                value_box=cell.box,
                confidence=round(confidence, 4),
                relationship=CandidateRelationship.TABLE_HEADER_TO_CELL,
                table_id=cell.table_id,
                row_index=cell.row_index,
                col_index=cell.col_index,
                evidence={
                    "value_match": value_score,
                    "header_match": header_score,
                    "row_index_match": request.row_index is None or request.row_index == cell.row_index,
                },
            )
        )
    return candidates


def _line_candidates(graph: DocumentGraph, request: FieldRequest) -> list[GroundingCandidate]:
    candidates: list[GroundingCandidate] = []
    words_by_id = graph.words_by_id()
    key_norm = normalize_text(request.key)
    for line in graph.lines:
        line_norm = normalize_text(line.text)
        if key_norm and key_norm not in line_norm:
            continue
        line_words = [words_by_id[word_id] for word_id in line.word_ids if word_id in words_by_id]
        indices, value_score = find_value_span(line_words, request.value, min_score=0.82)
        box = box_for_indices(line_words, indices) if indices else None
        if box is None:
            continue
        key_score = 1.0 if key_norm in line_norm else _ratio(line_norm, key_norm)
        confidence = 0.55 * value_score + 0.3 * key_score + 0.15
        candidates.append(
            GroundingCandidate(
                candidate_id=f"{request.field_id}:line:{line.line_id}",
                field_id=request.field_id,
                key=request.key,
                value=request.value,
                matched_text=" ".join(line_words[index].text for index in indices),
                page=line.page,
                value_box=box,
                key_box=line.box,
                confidence=round(confidence, 4),
                relationship=CandidateRelationship.SAME_LINE,
                evidence={"value_match": value_score, "key_match": key_score, "line_text": line.text},
            )
        )
    return candidates


def _value_only_candidates(graph: DocumentGraph, request: FieldRequest) -> list[GroundingCandidate]:
    indices, score = find_value_span(graph.words, request.value, min_score=0.9)
    box = box_for_indices(graph.words, indices) if indices else None
    if box is None:
        return []
    page = graph.words[indices[0]].page if indices else 0
    return [
        GroundingCandidate(
            candidate_id=f"{request.field_id}:value_only",
            field_id=request.field_id,
            key=request.key,
            value=request.value,
            matched_text=" ".join(graph.words[index].text for index in indices),
            page=page,
            value_box=box,
            confidence=round(score * 0.55, 4),
            relationship=CandidateRelationship.VALUE_ONLY,
            evidence={"value_match": score, "warning": "No key relationship found"},
        )
    ]


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()

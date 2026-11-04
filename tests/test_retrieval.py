from langchain_core.documents import Document

from backend.document_loader import MultiModalDocumentLoader
from backend.document_processor import DocumentProcessor


def _processor():
    return DocumentProcessor(MultiModalDocumentLoader())


def _doc(text):
    return Document(page_content=text)


def test_rrf_prefers_docs_found_by_both_retrievers():
    p = _processor()
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    semantic = [(a, 0.9), (b, 0.8)]
    bm25 = [(c, 5.0), (b, 4.0)]
    ranked = p.reciprocal_rank_fusion(semantic, bm25)
    assert ranked[0].page_content == "b"  # appears in both lists
    assert {d.page_content for d in ranked} == {"a", "b", "c"}


def test_rrf_deduplicates_and_handles_empty_inputs():
    p = _processor()
    d = _doc("same")
    assert len(p.reciprocal_rank_fusion([(d, 1)], [(d, 1)])) == 1
    assert p.reciprocal_rank_fusion([], []) == []

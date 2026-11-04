from backend.session_manager import SessionManager


def test_create_add_remove_document():
    sm = SessionManager()
    sid = sm.create_session()
    sm.add_document_to_session(sid, {"filename": "a.pdf", "chunks": 3})
    assert sm.get_session_info(sid)["total_chunks"] == 3
    doc_id = sm.get_session_documents(sid)[0]["document_id"]
    assert sm.remove_document_from_session(sid, doc_id)["filename"] == "a.pdf"
    assert sm.get_session_info(sid)["total_chunks"] == 0


def test_unknown_session_is_none():
    assert SessionManager().get_session_info("nope") is None

"""Upload -> index -> delete, fully offline (embeddings are stubbed)."""

TEXT = (
    "Quarterly revenue grew 12 percent driven by retail banking. "
    "Net interest margin was 3.4 percent. " * 40
).encode()


def _upload(client, name="report.txt", session_id=None):
    data = {"session_id": session_id} if session_id else {}
    return client.post("/api/upload", files={"file": (name, TEXT)}, data=data)


def test_upload_creates_session_and_indexes_chunks(client):
    r = _upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"]
    info = client.get(f"/api/session/{body['session_id']}").json()
    assert info["documents"][0]["filename"] == "report.txt"
    assert info["total_chunks"] > 0


def test_two_sessions_are_isolated(client):
    a = _upload(client, "a.txt").json()["session_id"]
    b = _upload(client, "b.txt").json()["session_id"]
    assert a != b
    names_a = [d["filename"] for d in client.get(f"/api/session/{a}").json()["documents"]]
    names_b = [d["filename"] for d in client.get(f"/api/session/{b}").json()["documents"]]
    assert names_a == ["a.txt"] and names_b == ["b.txt"]


def test_delete_document_then_end_session(client):
    sid = _upload(client).json()["session_id"]
    doc_id = client.get(f"/api/session/{sid}").json()["documents"][0]["document_id"]
    assert client.delete(f"/api/session/{sid}/document/{doc_id}").status_code == 200
    assert client.delete(f"/api/session/{sid}").status_code == 200
    assert client.get(f"/api/session/{sid}").status_code == 404


def test_ask_without_documents_is_rejected(client):
    sid = client.post("/api/start-session").json()["session_id"]
    r = client.post("/api/ask", json={"question": "hi", "session_id": sid})
    assert r.status_code == 400

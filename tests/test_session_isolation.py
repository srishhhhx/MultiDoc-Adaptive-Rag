def test_each_session_gets_its_own_workflow(api_module):
    a = api_module.get_session_workflow("session-a")
    b = api_module.get_session_workflow("session-b")
    assert a is not b
    assert api_module.get_session_workflow("session-a") is a
    assert a.current_session_id == "session-a"
    assert b.current_session_id == "session-b"


def test_retriever_state_does_not_leak_between_sessions(api_module):
    a = api_module.get_session_workflow("iso-a")
    b = api_module.get_session_workflow("iso-b")
    a.set_retriever(object())
    assert b.get_current_retriever() is None
    assert a.collection_name == "session_iso-a"


def test_drop_session_workflow(api_module):
    first = api_module.get_session_workflow("drop-me")
    api_module.drop_session_workflow("drop-me")
    assert api_module.get_session_workflow("drop-me") is not first

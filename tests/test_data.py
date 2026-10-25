from rag.data import split_queries

QUERIES = [{"id": str(i)} for i in range(100)]


def test_split_is_disjoint_complete_and_sized():
    dev, test = split_queries(QUERIES, dev_fraction=0.3)
    dev_ids, test_ids = {q["id"] for q in dev}, {q["id"] for q in test}
    assert len(dev) == 30 and len(test) == 70
    assert not dev_ids & test_ids
    assert dev_ids | test_ids == {q["id"] for q in QUERIES}


def test_split_is_deterministic_and_order_independent():
    assert split_queries(QUERIES) == split_queries(list(reversed(QUERIES)))

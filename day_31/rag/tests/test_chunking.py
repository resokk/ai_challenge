from conftest import FakeEmbedder


def test_chunks_follow_unit_boundaries_and_never_truncate():
    emb = FakeEmbedder(max_tokens=12)
    units = ["alpha beta gamma", "delta epsilon", " ".join(f"w{i}" for i in range(25)), "omega"]
    chunks = emb.chunk(units)
    assert chunks[0].text == "alpha beta gamma\ndelta epsilon"  # packed whole units
    split = [c for c in chunks if c.split_note]
    assert len(split) == 3  # 25 words, 10 per chunk -> explicit split, reported
    joined = " ".join(c.text for c in chunks).split()
    assert joined == " ".join(units).split()  # nothing dropped
    assert all(c.token_count <= 12 for c in chunks)

import pytest
from app.core import LEGAL_TAGS, format_clause_prompt, milestone_1, prepare_clause_dataset

def test_milestone_1():
    res = milestone_1()
    assert "train" in res and "test" in res
    assert res["total_samples"] >= 30
    assert res["train_samples"] > 0
    assert res["test_samples"] > 0
    for tag in LEGAL_TAGS:
        assert tag in res["tags"]
        assert res["tag_counts"][tag] > 0
    for item in res["train"] + res["test"]:
        assert "clause" in item and len(item["clause"]) > 10
        assert "tag" in item and item["tag"] in LEGAL_TAGS
        assert "prompt" in item
    assert format_clause_prompt("Sample.", "<TERMINATION>") == "Clause: Sample.\nTag: <TERMINATION>"
    assert format_clause_prompt("Sample.") == "Clause: Sample.\nTag:"

@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")

@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")

@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

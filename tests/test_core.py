import pytest
from transformers.pytorch_utils import Conv1D
from app.core import (
    LEGAL_TAGS, format_clause_prompt, milestone_1, prepare_clause_dataset,
    extend_tokenizer_and_model, milestone_2
)

def test_milestone_1():
    res = milestone_1()
    assert "train" in res and "test" in res
    assert res["total_samples"] >= 30

def test_milestone_2():
    res = milestone_2()
    model = res["model"]
    tokenizer = res["tokenizer"]
    custom_tags = res["custom_tags"]
    assert len(custom_tags) == len(LEGAL_TAGS)
    assert tokenizer.pad_token is not None
    for tag in custom_tags:
        token_id = tokenizer.convert_tokens_to_ids(tag)
        assert token_id is not None
        assert token_id < len(tokenizer)
    emb_weight = model.get_input_embeddings().weight
    assert emb_weight.shape[0] == len(tokenizer)

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

import pytest
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from app.core import (
    LEGAL_TAGS, format_clause_prompt, milestone_1, prepare_clause_dataset,
    extend_tokenizer_and_model, milestone_2,
    LoRAConv1D, LoRALinear, inject_lora, mark_only_lora_as_trainable, count_parameters, milestone_3
)

def test_milestone_1():
    res = milestone_1()
    assert "train" in res and "test" in res
    assert res["total_samples"] >= 30

def test_milestone_2():
    res = milestone_2()
    assert res["vocab_size"] == len(res["tokenizer"])

def test_milestone_3():
    base_conv = Conv1D(nf=64, nx=32)
    lora_conv = LoRAConv1D(base_conv, rank=4, alpha=8.0)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        assert torch.allclose(base_conv(x), lora_conv(x), atol=1e-6)
    assert not base_conv.weight.requires_grad
    assert lora_conv.lora_A.requires_grad
    res = milestone_3(rank=4, alpha=16.0)
    assert res["param_stats"]["trainable_params"] > 0

@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

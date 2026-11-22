import pytest
import torch
from transformers.pytorch_utils import Conv1D
from app.core import (
    LEGAL_TAGS, milestone_1, milestone_2, milestone_3, milestone_4,
    prepare_clause_dataset, extend_tokenizer_and_model, inject_lora,
    mark_only_lora_as_trainable, finetune_lora, LoRAConv1D
)

def test_milestone_1():
    assert milestone_1()["total_samples"] >= 30

def test_milestone_2():
    assert milestone_2()["vocab_size"] == len(milestone_2()["tokenizer"])

def test_milestone_3():
    base_conv = Conv1D(nf=64, nx=32)
    lora_conv = LoRAConv1D(base_conv, rank=4, alpha=8.0)
    assert lora_conv.lora_A.requires_grad

def test_milestone_4():
    dataset = prepare_clause_dataset(seed=42)
    small_train = dataset["train"][:8]
    model, tokenizer, _ = extend_tokenizer_and_model(is_small=True)
    inject_lora(model, rank=4, alpha=16.0)
    mark_only_lora_as_trainable(model)
    train_out = finetune_lora(model, tokenizer, small_train, epochs=2, batch_size=4, lr=2e-3, device="cpu")
    assert len(train_out["loss_history"]) == 2

@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

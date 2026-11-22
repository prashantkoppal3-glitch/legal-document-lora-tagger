"""Comprehensive tests for all 6 milestones of Legal Document LoRA Tagger."""

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from app.core import (
    LEGAL_TAGS,
    TAG_NAMES,
    LoRAConv1D,
    LoRALinear,
    count_parameters,
    evaluate_model_efficiency,
    evaluate_tagging_accuracy,
    extend_tokenizer_and_model,
    finetune_lora,
    format_clause_prompt,
    inject_lora,
    mark_only_lora_as_trainable,
    merge_lora_weights,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
    predict_clause_tag,
    prepare_clause_dataset,
    tag_document,
)


# --- Milestone 1: Prepare a dataset of legal clauses for tagging
def test_milestone_1():
    """Verify dataset loading, stratification, prompt formatting, and tag coverage."""
    res = milestone_1()

    assert "train" in res and "test" in res
    assert res["total_samples"] >= 30
    assert res["train_samples"] > 0
    assert res["test_samples"] > 0
    assert res["train_samples"] + res["test_samples"] == res["total_samples"]

    # Verify all expected legal tags are present in the dataset
    for tag in LEGAL_TAGS:
        assert tag in res["tags"]
        assert res["tag_counts"][tag] > 0

    # Verify prompt formatting in train and test items
    for item in res["train"] + res["test"]:
        assert "clause" in item and len(item["clause"]) > 10
        assert "tag" in item and item["tag"] in LEGAL_TAGS
        assert "prompt" in item
        assert item["prompt"].startswith("Clause: ")
        assert f"\nTag: {item['tag']}" in item["prompt"]

    # Test format_clause_prompt helper directly
    test_prompt = format_clause_prompt("Sample clause text.", "<TERMINATION>")
    assert test_prompt == "Clause: Sample clause text.\nTag: <TERMINATION>"

    inference_prompt = format_clause_prompt("Sample clause text.")
    assert inference_prompt == "Clause: Sample clause text.\nTag:"


# --- Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def test_milestone_2():
    """Verify tokenizer extension, special token IDs, and model embedding resizing."""
    res = milestone_2()

    model = res["model"]
    tokenizer = res["tokenizer"]
    custom_tags = res["custom_tags"]

    assert len(custom_tags) == len(LEGAL_TAGS)
    assert tokenizer.pad_token is not None

    # Check that each custom tag is present in the tokenizer vocabulary
    for tag in custom_tags:
        token_id = tokenizer.convert_tokens_to_ids(tag)
        assert token_id is not None
        assert token_id < len(tokenizer)
        # Check round-trip decoding
        decoded = tokenizer.decode([token_id])
        assert decoded == tag

    # Check model embedding matrix dimensions match extended tokenizer
    emb_weight = model.get_input_embeddings().weight
    assert emb_weight.shape[0] == len(tokenizer)
    assert res["vocab_size"] == len(tokenizer)
    assert res["embedding_size"] == emb_weight.shape[0]
    assert model.config.pad_token_id == tokenizer.pad_token_id


# --- Milestone 3: Implement LoRA adapter and inject into GPT-2
def test_milestone_3():
    """Verify LoRA module math, gradient freezing, and GPT-2 injection."""
    # 1. Test LoRAConv1D layer from scratch
    base_conv = Conv1D(nf=64, nx=32)
    lora_conv = LoRAConv1D(base_conv, rank=4, alpha=8.0)

    # Initial output should equal base output because lora_B is initialized to zeros
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        base_out = base_conv(x)
        lora_out = lora_conv(x)
    assert torch.allclose(base_out, lora_out, atol=1e-6)

    # Test parameter gradient requirements
    assert not base_conv.weight.requires_grad
    assert lora_conv.lora_A.requires_grad
    assert lora_conv.lora_B.requires_grad

    # 2. Test LoRALinear layer
    base_lin = nn.Linear(32, 64)
    lora_lin = LoRALinear(base_lin, rank=4, alpha=8.0)
    with torch.no_grad():
        lin_out = base_lin(x)
        lora_lin_out = lora_lin(x)
    assert torch.allclose(lin_out, lora_lin_out, atol=1e-6)

    # 3. Test milestone_3 helper and full model injection
    res = milestone_3(rank=4, alpha=16.0)
    model = res["model"]
    stats = res["param_stats"]

    assert stats["total_params"] > 0
    assert stats["trainable_params"] > 0
    assert stats["trainable_params"] < stats["total_params"]
    assert stats["trainable_pct"] < 10.0  # LoRA parameters should be a small fraction

    # Ensure all trainable parameters are exclusively LoRA parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name


# --- Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
def test_milestone_4():
    """Verify that fine-tuning optimizes LoRA parameters while keeping base weights frozen."""
    dataset = prepare_clause_dataset(seed=42)
    # Use small subset for quick unit testing
    small_train = dataset["train"][:8]

    model, tokenizer, _ = extend_tokenizer_and_model(is_small=True)
    inject_lora(model, rank=4, alpha=16.0)
    mark_only_lora_as_trainable(model)

    # Record initial weights of LoRA and base parameters
    lora_a_initial = None
    base_conv_initial = None
    for m in model.modules():
        if isinstance(m, LoRAConv1D):
            lora_a_initial = m.lora_A.detach().clone()
            base_conv_initial = m.base_layer.weight.detach().clone()
            break

    assert lora_a_initial is not None
    assert base_conv_initial is not None

    train_out = finetune_lora(
        model=model,
        tokenizer=tokenizer,
        train_data=small_train,
        epochs=2,
        batch_size=4,
        lr=2e-3,
        device="cpu",
    )

    assert len(train_out["loss_history"]) == 2
    assert train_out["final_loss"] is not None

    # Check that LoRA weights were updated
    for m in model.modules():
        if isinstance(m, LoRAConv1D):
            assert not torch.allclose(m.lora_A, lora_a_initial)
            # Base layer weights must remain strictly untouched
            assert torch.allclose(m.base_layer.weight, base_conv_initial)
            break


# --- Milestone 5: Merge LoRA adapter and perform inference on new clauses
def test_milestone_5():
    """Verify weight merging produces identical outputs to adapter and tests inference."""
    # 1. Test weight merging identity
    base_conv = Conv1D(nf=64, nx=32)
    lora_conv = LoRAConv1D(base_conv, rank=4, alpha=8.0)
    # Simulate updated LoRA weights
    with torch.no_grad():
        lora_conv.lora_B.copy_(torch.randn_like(lora_conv.lora_B) * 0.1)

    x = torch.randn(2, 4, 32)
    with torch.no_grad():
        out_before_merge = lora_conv(x)
        merged_conv = lora_conv.merge_weights()
        out_after_merge = merged_conv(x)

    assert torch.allclose(out_before_merge, out_after_merge, atol=1e-5)

    # 2. Test predict_clause_tag and tag_document
    model, tokenizer, _ = extend_tokenizer_and_model(is_small=True)
    sample_clause = "Either party may terminate this agreement upon 30 days written notice."
    pred = predict_clause_tag(model, tokenizer, sample_clause, candidate_tags=LEGAL_TAGS, device="cpu")

    assert "predicted_tag" in pred
    assert pred["predicted_tag"] in LEGAL_TAGS
    assert "confidence" in pred
    assert 0.0 <= pred["confidence"] <= 1.0

    # Test document tagging with multiple clauses
    doc = (
        "Either party may terminate this agreement.\n\n"
        "Both parties agree to keep all data confidential.\n\n"
        "This contract is governed by the laws of Texas."
    )
    tagged_doc = tag_document(model, tokenizer, doc, candidate_tags=LEGAL_TAGS)
    assert len(tagged_doc) == 3
    for clause_info in tagged_doc:
        assert clause_info["predicted_tag"] in LEGAL_TAGS
        assert "clause_index" in clause_info


# --- Milestone 6: Evaluate tagging accuracy and model efficiency
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    """Verify evaluation metrics: accuracy, precision, recall, F1, and latency benchmarking."""
    model, tokenizer, _ = extend_tokenizer_and_model(is_small=True)

    test_clauses = [
        {"clause": "Either party may terminate upon written notice.", "tag": "<TERMINATION>"},
        {"clause": "All confidential information shall be protected.", "tag": "<CONFIDENTIALITY>"},
        {"clause": "Governed by Delaware state law.", "tag": "<GOVERNING_LAW>"},
        {"clause": "Contractor shall indemnify and hold harmless the city.", "tag": "<INDEMNIFICATION>"},
        {"clause": "Neither party liable for acts of God or war.", "tag": "<FORCE_MAJEURE>"},
    ]

    # Evaluate accuracy
    metrics = evaluate_tagging_accuracy(model, tokenizer, test_clauses, tags=LEGAL_TAGS)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    for tag in LEGAL_TAGS:
        assert tag in metrics["per_class"]
        assert "f1" in metrics["per_class"][tag]

    # Evaluate efficiency
    eval_clauses = [c["clause"] for c in test_clauses]
    efficiency = evaluate_model_efficiency(model, tokenizer, eval_clauses, rank=4, device="cpu")
    assert "avg_latency_ms" in efficiency
    assert efficiency["avg_latency_ms"] >= 0.0
    assert "throughput_clauses_per_sec" in efficiency
    assert "total_parameters" in efficiency
    assert efficiency["estimated_lora_trainable_pct"] < 5.0

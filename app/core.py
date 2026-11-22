"""Core implementation for Legal Document LoRA Tagger.

Public sector clerks classify legal documents and public records faster by fine-tuning
GPT-2 with custom LoRA adapters to tag clauses accurately and efficiently.
"""

from __future__ import annotations

import copy
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D


# ---------------------------------------------------------------------------
# Default Tags & Dataset Definitions
# ---------------------------------------------------------------------------

LEGAL_TAGS: List[str] = [
    "<TERMINATION>",
    "<CONFIDENTIALITY>",
    "<GOVERNING_LAW>",
    "<INDEMNIFICATION>",
    "<FORCE_MAJEURE>",
]

TAG_NAMES = {
    "<TERMINATION>": "Termination",
    "<CONFIDENTIALITY>": "Confidentiality",
    "<GOVERNING_LAW>": "Governing Law",
    "<INDEMNIFICATION>": "Indemnification",
    "<FORCE_MAJEURE>": "Force Majeure",
}

# Curated benchmark dataset of diverse clauses found in public contracts and records
RAW_CLAUSES_CORPUS: List[Dict[str, str]] = [
    # Termination clauses
    {
        "clause": "Either party may terminate this Agreement without cause by providing at least thirty (30) days prior written notice to the other party.",
        "tag": "<TERMINATION>",
    },
    {
        "clause": "Upon material breach by either party of any obligation hereunder, the non-breaching party may terminate this Agreement immediately if such breach remains uncured for fifteen (15) days following written notice.",
        "tag": "<TERMINATION>",
    },
    {
        "clause": "This contract shall automatically terminate upon the bankruptcy, insolvency, or liquidation of either contracting entity.",
        "tag": "<TERMINATION>",
    },
    {
        "clause": "The public agency reserves the right to cancel or terminate this procurement contract immediately in the event of allocated fund deauthorization.",
        "tag": "<TERMINATION>",
    },
    {
        "clause": "In the event of convenience termination by the government entity, contractor shall be compensated solely for verifiable work completed prior to notice.",
        "tag": "<TERMINATION>",
    },
    {
        "clause": "Failure to meet statutory service delivery milestones within forty-five (45) days shall entitle the department to terminate this grant agreement.",
        "tag": "<TERMINATION>",
    },

    # Confidentiality clauses
    {
        "clause": "The receiving party agrees to safeguard and maintain the confidentiality of all proprietary technical, financial, and operational information disclosed by the disclosing party.",
        "tag": "<CONFIDENTIALITY>",
    },
    {
        "clause": "Confidential Information shall not include data that is or becomes part of the public domain through no fault or breach of the receiving party.",
        "tag": "<CONFIDENTIALITY>",
    },
    {
        "clause": "Both parties agree that non-public citizen records and sealed agency data shall remain strictly confidential for a minimum period of five (5) years.",
        "tag": "<CONFIDENTIALITY>",
    },
    {
        "clause": "Recipient shall limit access to Confidential Materials solely to authorized employees having a legitimate need-to-know under binding non-disclosure terms.",
        "tag": "<CONFIDENTIALITY>",
    },
    {
        "clause": "Any unauthorized disclosure or breach of trade secrets shall cause irreparable harm entitling the disclosing party to immediate injunctive relief.",
        "tag": "<CONFIDENTIALITY>",
    },
    {
        "clause": "Upon conclusion of the contract engagement, all copies of agency confidential data and backups must be certified destroyed or returned within ten (10) days.",
        "tag": "<CONFIDENTIALITY>",
    },

    # Governing Law clauses
    {
        "clause": "This Agreement and all disputes arising hereunder shall be governed by, and construed in accordance with, the laws of the State of New York, without regard to conflict of law principles.",
        "tag": "<GOVERNING_LAW>",
    },
    {
        "clause": "Any legal suit, action, or proceeding arising out of or related to this public sector agreement shall be instituted exclusively in the federal or state courts located in Travis County, Texas.",
        "tag": "<GOVERNING_LAW>",
    },
    {
        "clause": "The validity, interpretation, and performance of this procurement shall be determined under the administrative rules and civil statutes of the Commonwealth of Pennsylvania.",
        "tag": "<GOVERNING_LAW>",
    },
    {
        "clause": "All claims or controversies between the vendor and the municipality shall be resolved under the statutory jurisdiction of the State of California municipal courts.",
        "tag": "<GOVERNING_LAW>",
    },
    {
        "clause": "This contract shall be construed in accordance with the laws of the State of Delaware, and the parties submit to the exclusive jurisdiction of the Delaware Court of Chancery.",
        "tag": "<GOVERNING_LAW>",
    },
    {
        "clause": "Applicable federal procurement standards and the laws of the District of Columbia shall govern the adjudication of any claim hereunder.",
        "tag": "<GOVERNING_LAW>",
    },

    # Indemnification clauses
    {
        "clause": "Contractor shall defend, indemnify, and hold harmless the City, its officers, employees, and agents from any and all liabilities, losses, damages, or claims arising from contractor negligence.",
        "tag": "<INDEMNIFICATION>",
    },
    {
        "clause": "The vendor agrees to indemnify the public agency against all third-party intellectual property infringement claims resulting from software deliverables furnished hereunder.",
        "tag": "<INDEMNIFICATION>",
    },
    {
        "clause": "Each party shall indemnify and hold the other party harmless against any property damage, bodily injury, or statutory fines caused by its gross negligence or willful misconduct.",
        "tag": "<INDEMNIFICATION>",
    },
    {
        "clause": "The service provider assumes full liability and agrees to defend and indemnify the state against any data breach penalties resulting from non-compliance with security standards.",
        "tag": "<INDEMNIFICATION>",
    },
    {
        "clause": "Indemnification obligations under this section shall survive the expiration, termination, or cancellation of this legal agreement.",
        "tag": "<INDEMNIFICATION>",
    },
    {
        "clause": "The contractor shall hold harmless the local township against all worker compensation claims, subcontractor liens, and associated legal defense expenses.",
        "tag": "<INDEMNIFICATION>",
    },

    # Force Majeure clauses
    {
        "clause": "Neither party shall be held liable for failure or delay in performing its obligations under this contract if such failure arises from acts of God, floods, fires, or pandemics beyond reasonable control.",
        "tag": "<FORCE_MAJEURE>",
    },
    {
        "clause": "If an event of force majeure, including war, civil unrest, or state of emergency, persists for more than sixty (60) continuous days, either party may terminate the agreement without penalty.",
        "tag": "<FORCE_MAJEURE>",
    },
    {
        "clause": "The affected party must provide prompt written notice within seventy-two (72) hours of the onset of an excusable delay event detailing anticipated duration and remediation steps.",
        "tag": "<FORCE_MAJEURE>",
    },
    {
        "clause": "Labor strikes, national supply chain disruptions, or severe weather events hindering transport shall suspend performance requirements for the duration of the disruption.",
        "tag": "<FORCE_MAJEURE>",
    },
    {
        "clause": "Performance deadlines shall be extended by an amount of time equal to the excusable delay caused directly by government shutdown or sovereign embargo actions.",
        "tag": "<FORCE_MAJEURE>",
    },
    {
        "clause": "Obligations to pay accrued monetary fees for prior delivered services shall not be excused or suspended by reason of a force majeure occurrence.",
        "tag": "<FORCE_MAJEURE>",
    },
]


def format_clause_prompt(clause: str, tag: Optional[str] = None) -> str:
    """Format a clause and optional tag into the training / inference prompt template."""
    clean_clause = clause.strip()
    if tag:
        return f"Clause: {clean_clause}\nTag: {tag}"
    return f"Clause: {clean_clause}\nTag:"


# --- Milestone 1: Prepare a dataset of legal clauses for tagging
def prepare_clause_dataset(
    corpus: Optional[List[Dict[str, str]]] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    as_hf_dataset: bool = False,
) -> Dict[str, Any]:
    """Prepare and split a dataset of legal clauses for tagging.

    Args:
        corpus: List of dicts with 'clause' and 'tag' keys.
        test_ratio: Proportion of clauses to allocate to test set.
        seed: Random seed for deterministic reproducibility.
        as_hf_dataset: If True, also return Hugging Face Dataset objects.

    Returns:
        Dictionary containing train and test clause lists, tag distribution, and formatted samples.
    """
    raw_data = list(corpus or RAW_CLAUSES_CORPUS)
    import random

    rng = random.Random(seed)
    # Stratified split to ensure balanced representation across tags
    by_tag: Dict[str, List[Dict[str, str]]] = {}
    for item in raw_data:
        by_tag.setdefault(item["tag"], []).append(item)

    train_data: List[Dict[str, str]] = []
    test_data: List[Dict[str, str]] = []

    for tag, items in by_tag.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        test_data.extend(shuffled[:n_test])
        train_data.extend(shuffled[n_test:])

    # Format training prompts
    for d in train_data:
        d["prompt"] = format_clause_prompt(d["clause"], d["tag"])
    for d in test_data:
        d["prompt"] = format_clause_prompt(d["clause"], d["tag"])

    tag_counts = {t: len(items) for t, items in by_tag.items()}

    result: Dict[str, Any] = {
        "train": train_data,
        "test": test_data,
        "total_samples": len(raw_data),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "tags": sorted(list(by_tag.keys())),
        "tag_counts": tag_counts,
    }

    if as_hf_dataset:
        try:
            from datasets import Dataset, DatasetDict

            result["hf_dataset"] = DatasetDict({
                "train": Dataset.from_list(train_data),
                "test": Dataset.from_list(test_data),
            })
        except Exception:
            result["hf_dataset"] = None

    return result


def milestone_1() -> Dict[str, Any]:
    """Milestone 1: Prepare a dataset of legal clauses for tagging."""
    dataset = prepare_clause_dataset(as_hf_dataset=True)
    return dataset


# --- Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def extend_tokenizer_and_model(
    model_name_or_cfg: Union[str, GPT2Config] = "gpt2",
    custom_tags: Optional[List[str]] = None,
    is_small: bool = True,
    local_files_only: bool = True,
) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer, List[str]]:
    """Extend the GPT-2 tokenizer with custom tags and resize model embeddings.

    Args:
        model_name_or_cfg: Model name or custom GPT2Config instance.
        custom_tags: List of custom special tags to add.
        is_small: If True and initializing from config, use compact dimensions for rapid testing.
        local_files_only: Whether to load offline from local cache.

    Returns:
        Tuple of (extended model, extended tokenizer, list of added tags).
    """
    tags_to_add = custom_tags or list(LEGAL_TAGS)

    # Load tokenizer from local cache or default
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=local_files_only)
    except Exception:
        # Fallback if local_files_only needs False or offline loading
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set padding token to EOS token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add custom legal tags as special tokens
    added_count = tokenizer.add_special_tokens({"additional_special_tokens": tags_to_add})

    # Instantiate or load model
    if isinstance(model_name_or_cfg, GPT2Config):
        cfg = model_name_or_cfg
        model = GPT2LMHeadModel(cfg)
    elif is_small:
        # Fast, lightweight GPT-2 config suitable for CI and low-latency testing
        cfg = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_embd=128,
            n_layer=2,
            n_head=2,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPT2LMHeadModel(cfg)
    else:
        try:
            model = GPT2LMHeadModel.from_pretrained(model_name_or_cfg, local_files_only=local_files_only)
        except Exception:
            cfg = GPT2Config.from_pretrained("gpt2", local_files_only=local_files_only)
            model = GPT2LMHeadModel(cfg)

    # Resize model token embeddings to match new tokenizer vocabulary
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, tags_to_add


def milestone_2() -> Dict[str, Any]:
    """Milestone 2: Extend GPT-2 tokenizer and model for custom tags."""
    model, tokenizer, tags = extend_tokenizer_and_model(is_small=True)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "custom_tags": tags,
        "vocab_size": len(tokenizer),
        "embedding_size": model.get_input_embeddings().weight.shape[0],
    }


# --- Milestone 3: Implement LoRA adapter and inject into GPT-2
class LoRAConv1D(nn.Module):
    """LoRA wrapper for GPT-2 Conv1D projection layers built from scratch.

    GPT-2 attention projections (c_attn, c_proj) store weights with shape
    (in_features, out_features). LoRA introduces low-rank matrices:
        A: (in_features, rank)
        B: (rank, out_features)
    such that delta_W = (A @ B) * (alpha / rank).
    """

    def __init__(self, base_layer: Conv1D, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze the pre-trained base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.weight.shape[0]
        out_features = base_layer.weight.shape[1]
        self.in_features = in_features
        self.out_features = out_features

        if rank > 0:
            # Low-rank decomposition matrices
            self.lora_A = nn.Parameter(torch.empty(in_features, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

            # Initialize A with Kaiming uniform and B with zeros so delta_W starts at 0
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base Conv1D output
        base_out = self.base_layer(x)

        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            # Low-rank adapted output: (x @ A) @ B * scaling
            lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
            return base_out + lora_out

        return base_out

    def merge_weights(self) -> Conv1D:
        """Merge low-rank weights directly into base Conv1D weights and return unwrapped layer."""
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.base_layer.weight.data += delta_w
        return self.base_layer


class LoRALinear(nn.Module):
    """LoRA wrapper for standard nn.Linear layers built from scratch.

    Linear stores weights with shape (out_features, in_features).
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.in_features = in_features
        self.out_features = out_features

        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
            return base_out + lora_out
        return base_out

    def merge_weights(self) -> nn.Linear:
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
        return self.base_layer


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 16.0,
    target_modules: Tuple[str, ...] = ("c_attn", "c_proj"),
) -> nn.Module:
    """Recursively inject custom LoRA adapters into GPT-2 attention layers."""
    for name, child in list(model.named_children()):
        if isinstance(child, Conv1D) and any(tgt in name for tgt in target_modules):
            setattr(model, name, LoRAConv1D(child, rank=rank, alpha=alpha))
        elif isinstance(child, nn.Linear) and any(tgt in name for tgt in target_modules):
            setattr(model, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            inject_lora(child, rank=rank, alpha=alpha, target_modules=target_modules)
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all base parameters, leaving only LoRA parameters trainable."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Calculate parameter counts and compression statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_pct": round(trainable_pct, 4),
    }


def milestone_3(
    model: Optional[nn.Module] = None,
    rank: int = 4,
    alpha: float = 16.0,
) -> Dict[str, Any]:
    """Milestone 3: Implement LoRA adapter and inject into GPT-2."""
    if model is None:
        model, _, _ = extend_tokenizer_and_model(is_small=True)

    inject_lora(model, rank=rank, alpha=alpha)
    mark_only_lora_as_trainable(model)
    param_stats = count_parameters(model)

    return {
        "model": model,
        "rank": rank,
        "alpha": alpha,
        "param_stats": param_stats,
    }


# --- Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
def tokenize_clause_data(
    data: List[Dict[str, str]],
    tokenizer: GPT2Tokenizer,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """Tokenize clauses and tags for causal language modeling."""
    prompts = [item.get("prompt") or format_clause_prompt(item["clause"], item["tag"]) for item in data]
    # Append EOS token to delimit sequence completion
    full_texts = [p + tokenizer.eos_token for p in prompts]

    encodings = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    labels = encodings["input_ids"].clone()
    # Mask padding tokens in the loss computation
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


def finetune_lora(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    train_data: List[Dict[str, str]],
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Fine-tune LoRA parameters on the legal clause dataset using PyTorch.

    Args:
        model: GPT-2 model with injected LoRA adapters.
        tokenizer: Extended GPT-2 tokenizer.
        train_data: List of clause-tag dictionaries.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for AdamW optimizer.
        device: Device to train on ('cuda' or 'cpu').

    Returns:
        Dictionary with training loss history, final loss, and training duration.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    # Ensure only LoRA parameters receive gradient updates
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in model. Call mark_only_lora_as_trainable first.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    tokenized = tokenize_clause_data(train_data, tokenizer)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = tokenized["labels"]

    n_samples = input_ids.shape[0]
    loss_history: List[float] = []
    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle indices each epoch
        perm = torch.randperm(n_samples)
        epoch_losses: List[float] = []

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            b_input_ids = input_ids[batch_idx].to(device)
            b_attention_mask = attention_mask[batch_idx].to(device)
            b_labels = labels[batch_idx].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        loss_history.append(avg_epoch_loss)

    train_duration = time.time() - start_time
    model.eval()

    return {
        "model": model,
        "epochs": epochs,
        "loss_history": loss_history,
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "duration_seconds": round(train_duration, 3),
    }


def milestone_4(
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset."""
    dataset = prepare_clause_dataset()
    model, tokenizer, _ = extend_tokenizer_and_model(is_small=True)
    inject_lora(model, rank=4, alpha=16.0)
    mark_only_lora_as_trainable(model)

    train_results = finetune_lora(
        model=model,
        tokenizer=tokenizer,
        train_data=dataset["train"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    return {
        "train_results": train_results,
        "tokenizer": tokenizer,
        "model": train_results["model"],
    }


# --- Milestone 5: Merge LoRA adapter and perform inference on new clauses
def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")

def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")

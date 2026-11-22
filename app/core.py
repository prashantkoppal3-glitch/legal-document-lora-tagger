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
def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")

def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")

def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")

def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")

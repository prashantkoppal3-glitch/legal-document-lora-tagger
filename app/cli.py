"""Command-line entry point: `python -m app.cli --help`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from app.core import (
    LEGAL_TAGS,
    TAG_NAMES,
    count_parameters,
    evaluate_model_efficiency,
    evaluate_tagging_accuracy,
    extend_tokenizer_and_model,
    finetune_lora,
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="legal-document-lora-tagger",
        description="Public sector clerks classify documents faster. This tool uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, improving efficiency and reducing manual review time.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Hello command
    sub.add_parser("hello", help="check that everything is wired up")

    # Milestone 1: Prepare data
    p_data = sub.add_parser("prepare-data", help="prepare and inspect legal clause dataset")
    p_data.add_argument("--test-ratio", type=float, default=0.2, help="proportion of dataset for evaluation")

    # Milestone 2: Extend tokenizer and model
    p_setup = sub.add_parser("setup-model", help="extend GPT-2 tokenizer and model embeddings with legal tags")

    # Milestone 3: Inject LoRA
    p_lora = sub.add_parser("inject-lora", help="inject LoRA adapters into GPT-2 attention projections")
    p_lora.add_argument("--rank", type=int, default=4, help="LoRA rank dimension")
    p_lora.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha scaling factor")

    # Milestone 4: Train LoRA model
    p_train = sub.add_parser("train", help="fine-tune GPT-2 LoRA adapters on legal clause dataset")
    p_train.add_argument("--epochs", type=int, default=3, help="number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=4, help="mini-batch size")
    p_train.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    # Milestone 5: Predict clause or tag document
    p_pred = sub.add_parser("predict", help="classify a single legal clause")
    p_pred.add_argument("clause", type=str, help="text of the legal clause to classify")

    p_doc = sub.add_parser("tag-doc", help="tag all clauses within a legal document text")
    p_doc.add_argument("document", type=str, help="raw multi-clause contract or document text")

    # Milestone 6: Evaluate
    p_eval = sub.add_parser("evaluate", help="evaluate tagging accuracy and model efficiency")
    p_eval.add_argument("--rank", type=int, default=4, help="LoRA rank to evaluate")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")

    elif args.command == "prepare-data":
        data = prepare_clause_dataset(test_ratio=args.test_ratio)
        print("Dataset prepared successfully:")
        print(f"  Total samples: {data['total_samples']}")
        print(f"  Train samples: {data['train_samples']}")
        print(f"  Test samples:  {data['test_samples']}")
        print("  Tag distribution:")
        for tag, count in data["tag_counts"].items():
            print(f"    {tag} ({TAG_NAMES.get(tag, '')}): {count}")

    elif args.command == "setup-model":
        model, tokenizer, tags = extend_tokenizer_and_model(is_small=True)
        print("Extended GPT-2 with legal tags:")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"  Added special tags: {', '.join(tags)}")
        print(f"  Embedding shape: {tuple(model.get_input_embeddings().weight.shape)}")

    elif args.command == "inject-lora":
        res = milestone_3(rank=args.rank, alpha=args.alpha)
        stats = res["param_stats"]
        print(f"LoRA injected into GPT-2 attention layers (Rank={args.rank}, Alpha={args.alpha}):")
        print(f"  Total parameters:     {stats['total_params']:,}")
        print(f"  Trainable parameters: {stats['trainable_params']:,} ({stats['trainable_pct']}%)")
        print(f"  Frozen parameters:    {stats['frozen_params']:,}")

    elif args.command == "train":
        print(f"Fine-tuning GPT-2 with LoRA (Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr})...")
        res = milestone_4(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        train_res = res["train_results"]
        print("Training complete:")
        for ep, loss in enumerate(train_res["loss_history"], start=1):
            print(f"  Epoch {ep}/{args.epochs} - Average Loss: {loss:.4f}")
        print(f"  Initial Loss: {train_res['initial_loss']:.4f}")
        print(f"  Final Loss:   {train_res['final_loss']:.4f}")
        print(f"  Duration:     {train_res['duration_seconds']}s")

    elif args.command == "predict":
        model, tokenizer, _ = extend_tokenizer_and_model(is_small=True)
        inject_lora(model, rank=4, alpha=16.0)
        mark_only_lora_as_trainable(model)
        dataset = prepare_clause_dataset()
        finetune_lora(model, tokenizer, dataset["train"], epochs=2)
        merge_lora_weights(model)

        pred = predict_clause_tag(model, tokenizer, args.clause)
        print(f"Clause: {pred['clause']}")
        print(f"Predicted Tag: {pred['predicted_tag']} ({pred['tag_name']})")
        print(f"Confidence:    {pred['confidence'] * 100:.2f}%")
        print("Candidate probabilities:")
        for t, prob in pred["tag_probabilities"].items():
            print(f"  {t} ({TAG_NAMES.get(t, '')}): {prob * 100:.2f}%")

    elif args.command == "tag-doc":
        model, tokenizer, _ = extend_tokenizer_and_model(is_small=True)
        inject_lora(model, rank=4, alpha=16.0)
        mark_only_lora_as_trainable(model)
        dataset = prepare_clause_dataset()
        finetune_lora(model, tokenizer, dataset["train"], epochs=2)
        merge_lora_weights(model)

        tagged = tag_document(model, tokenizer, args.document)
        print(f"Tagged {len(tagged)} clauses in legal document:")
        for t in tagged:
            print(f"\n[Clause #{t['clause_index']}] Tag: {t['predicted_tag']} ({t['tag_name']}) - Confidence: {t['confidence']*100:.1f}%")
            print(f"Text: {t['clause']}")

    elif args.command == "evaluate":
        print("Evaluating model accuracy and efficiency...")
        res = milestone_6()
        acc = res["accuracy_metrics"]
        eff = res["efficiency_metrics"]
        print("\n=== Accuracy Metrics ===")
        print(f"  Overall Accuracy: {acc['accuracy'] * 100:.2f}%")
        print(f"  Macro F1 Score:   {acc['macro_f1']:.4f}")
        print("  Per-class Breakdown:")
        for tag, m in acc["per_class"].items():
            print(f"    {tag} ({m['name']}): Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f} (n={m['support']})")

        print("\n=== Efficiency Metrics ===")
        print(f"  Average Inference Latency: {eff['avg_latency_ms']} ms")
        print(f"  Throughput:                {eff['throughput_clauses_per_sec']} clauses/sec")
        print(f"  Total Model Parameters:    {eff['total_parameters']:,}")
        print(f"  Estimated LoRA Footprint:  {eff['estimated_lora_trainable_pct']}% of base model")


if __name__ == "__main__":
    main()

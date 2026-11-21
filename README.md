# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/sza6389ebb/badge.svg)](https://ask.thili.ai/b/sza6389ebb)

This project develops a Legal Document LoRA Tagger, designed to assist public sector clerks in classifying legal clauses more efficiently. By fine-tuning GPT-2 with a LoRA adapter, the tool automates the tagging of legal clauses in public records, significantly reducing manual review time and improving operational efficiency.

## What you are building

**Goal.** You will build a tool that fine-tunes GPT-2 using LoRA to classify legal clauses in public records, resulting in a tagged legal document.

**Done when.** You will have a fine-tuned GPT-2 model with a LoRA adapter that can classify legal contract clauses by type, and you will have evaluated its tagging accuracy and efficiency on new legal documents.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a dataset of legal clauses for tagging**  
      Hint: Use techniques from 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' to prepare your dataset.
- [ ] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Hint: Apply the special token extension method from 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' to your tokenizer and model.
- [ ] **3. Implement LoRA adapter and inject into GPT-2**  
      Hint: Implement and inject the LoRA adapter into GPT-2's attention layers, as shown in 'Build LoRA from Scratch'.
- [ ] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Hint: Combine the LoRA adapter technique from 'Build LoRA from Scratch' with the fine-tuning process from 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.
- [ ] **5. Merge LoRA adapter and perform inference on new clauses**  
      Hint: Utilize the LoRA adapter merging and inference techniques from 'Build LoRA from Scratch' and 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.
- [ ] **6. Evaluate tagging accuracy and model efficiency**  
      Hint: Evaluate the model's performance using metrics similar to those discussed in 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different LoRA ranks and alpha values to observe their impact on tagging accuracy and training efficiency.
- Perform a detailed error analysis on misclassified clauses to identify patterns and suggest potential dataset or model improvements.
- Compare the fine-tuning speed and memory footprint of your LoRA implementation against the theoretical resource requirements of full GPT-2 fine-tuning.

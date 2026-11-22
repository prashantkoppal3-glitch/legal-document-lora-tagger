# Legal Document LoRA Tagger

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/sza6389ebb/badge.svg)](https://msrit.ask.thili.ai/b/sza6389ebb)
![Tests](https://github.com/prashantkoppal3-glitch/legal-document-lora-tagger/actions/workflows/ci.yml/badge.svg)

This project develops a **Legal Document LoRA Tagger**, designed to assist public sector clerks and legal reviewers in classifying legal clauses more efficiently. By fine-tuning GPT-2 with a custom **LoRA (Low-Rank Adaptation)** adapter built from scratch, the tool automates the classification and tagging of legal clauses in public records, significantly reducing manual review time while keeping parameter footprint minimal (over 99.8% parameter reduction).

---

## What you are building

**Goal.** Build a tool that fine-tunes GPT-2 using a scratch-built LoRA adapter to classify and tag legal clauses in public records.

**Done when.** You have a fine-tuned GPT-2 model with a LoRA adapter that can classify legal contract clauses by type, merge weights with zero inference latency overhead, and achieve verified tagging accuracy and efficiency on new legal documents.

---

## Milestones

- [x] **1. Prepare a dataset of legal clauses for tagging**  
      Curated stratified public records dataset across 5 core legal clause types (`<TERMINATION>`, `<CONFIDENTIALITY>`, `<GOVERNING_LAW>`, `<INDEMNIFICATION>`, `<FORCE_MAJEURE>`) with prompt templates.
- [x] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Extended vocabulary with special tokens, resized model embeddings, and aligned padding tokens.
- [x] **3. Implement LoRA adapter and inject into GPT-2**  
      Built custom `LoRAConv1D` and `LoRALinear` layers from scratch in PyTorch, injected them into attention projections (`c_attn`, `c_proj`), and froze 99.84% of base weights.
- [x] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Causal language modeling training loop with AdamW optimizer updating only low-rank matrices.
- [x] **5. Merge LoRA adapter and perform inference on new clauses**  
      Merged $\Delta W$ directly into base pre-trained weights for zero-overhead inference, supporting single-clause prediction and full document segmentation/tagging.
- [x] **6. Evaluate tagging accuracy and model efficiency**  
      Comprehensive evaluation suite covering precision, recall, per-class & macro F1, parameter reduction, and inference latency.

---

## Quick Start & Installation

### Prerequisites
- Python 3.11 or 3.12
- Git

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/prashantkoppal3-glitch/legal-document-lora-tagger.git
cd legal-document-lora-tagger

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# or: .venv\Scripts\activate    # On Windows PowerShell
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```
*(Optionally set an LLM provider in `.env` if you wish to use the generic LLM chat helper. The core LoRA tagger runs completely offline and locally without requiring any external API keys).*

### 4. Run the Test Suite

```bash
pytest -v
```

All 9 tests (6 milestone unit/integration tests and 3 smoke tests) pass out-of-the-box:
```
tests/test_core.py::test_milestone_1 PASSED
tests/test_core.py::test_milestone_2 PASSED
tests/test_core.py::test_milestone_3 PASSED
tests/test_core.py::test_milestone_4 PASSED
tests/test_core.py::test_milestone_5 PASSED
tests/test_core.py::test_milestone_6 PASSED
tests/test_smoke.py::test_the_store_keeps_what_you_put_in_it PASSED
tests/test_smoke.py::test_model_settings_default_to_a_local_ollama PASSED
tests/test_smoke.py::test_the_command_line_runs PASSED
========================= 9 passed in 15.79s =========================
```

---

## CLI Usage

The project provides an entry point at `app.cli` to execute every phase of the workflow:

```bash
# 1. Verify CLI plumbing
python -m app.cli hello

# 2. Prepare and inspect the legal clauses dataset
python -m app.cli prepare-data

# 3. Extend GPT-2 tokenizer and inspect resized embeddings
python -m app.cli setup-model

# 4. Inject LoRA adapters and display parameter efficiency
python -m app.cli inject-lora --rank 4 --alpha 16.0

# 5. Fine-tune LoRA on the legal clauses dataset
python -m app.cli train --epochs 3 --batch-size 4 --lr 0.001

# 6. Predict the tag for an unseen legal clause
python -m app.cli predict "Either party may terminate this agreement upon 30 days written notice."

# 7. Tag all clauses in a multi-clause contract
python -m app.cli tag-doc "In the event of breach, party may terminate.\n\nAll confidential data shall remain private."

# 8. Evaluate tagging accuracy and model latency
python -m app.cli evaluate
```

---

## Technical Architecture

### 1. LoRA From Scratch
In standard fine-tuning, every weight parameter $W_0 \in \mathbb{R}^{d_{in} \times d_{out}}$ accumulates gradients. LoRA freezes $W_0$ and parameterizes the update as a low-rank product:

$$\Delta W = (A \cdot B) \cdot \frac{\alpha}{r}$$

- $A \in \mathbb{R}^{d_{in} \times r}$ initialized with Kaiming uniform
- $B \in \mathbb{R}^{r \times d_{out}}$ initialized to zero (ensuring $\Delta W = 0$ at start of training)
- Rank $r = 4$, scaling $\alpha = 16.0$

### 2. Attention Layer Replacement
Hugging Face's GPT-2 utilizes `Conv1D` projection layers with transposed weight dimensions $(d_{in}, d_{out})$. Our custom `LoRAConv1D` module wraps these layers directly, applying $(x \cdot A) \cdot B \cdot \text{scaling}$ and preserving complete architectural compatibility.

### 3. Weight Merging
For deployment, $\Delta W$ is directly added back to the base layer weights:
$$W_{merged} = W_0 + \Delta W$$
This completely removes the LoRA module during inference, guaranteeing zero latency overhead compared to the base model.

---

## Project Structure

```
legal-document-lora-tagger/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI test workflow
├── app/
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface entry point
│   ├── core.py                  # Milestones 1-6 core implementations
│   ├── llm.py                   # Optional chat completions interface
│   ├── storage.py               # Key-value JSON datastore
│   └── tokenizer/               # Bundled offline GPT-2 tokenizer assets
├── tests/
│   ├── test_core.py             # Milestone test suite
│   └── test_smoke.py            # Baseline plumbing smoke tests
├── .env.example                 # Example configuration
├── .gitignore                   # Git ignore file (excludes .env)
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## Settings and Secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever reaches GitHub, treat it as leaked and generate a new one immediately.

---

## License

This project is licensed under the MIT License.

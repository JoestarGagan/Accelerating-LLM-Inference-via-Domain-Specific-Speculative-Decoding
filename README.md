# Accelerating LLM Inference via Domain-Specific Speculative Decoding

## Abstract
This project investigates the latency optimization of Large Language Models (LLMs) through **Domain-Specific Speculative Decoding**. While standard speculative decoding utilizes generic draft models to accelerate inference, this research hypothesizes that aligning the draft model's distribution with the target domain (specifically Python programming) significantly increases the **Token Acceptance Rate (TAR)**. By fine-tuning a compact draft model (1B parameters) on domain-specific corpora, this system aims to achieve higher effective throughput and reduced wall-clock latency compared to generic baselines, without compromising the generation quality of the target model (8B parameters).

## 1. Introduction
Large Language Models are predominantly memory-bound during inference due to the auto-regressive nature of token generation. Speculative Decoding addresses this bottleneck by leveraging a smaller, faster "draft model" to generate candidate token sequences, which are subsequently verified in parallel by the larger "target model."

The efficiency of this pipeline is strictly governed by the **Acceptance Rate ($\alpha$)**—the probability that the target model validates the draft tokens. Generic draft models often exhibit low $\alpha$ on specialized tasks, such as code generation or biomedical analysis, leading to suboptimal speedups. This project proposes a **Domain-Adaptive Speculative Decoding** framework to maximize $\alpha$ through supervised fine-tuning (SFT) of the drafter.

## 2. System Architecture

The architecture consists of two distinct phases: **Offline Alignment** and **Online Speculative Inference**.

### 2.1 Offline Alignment
The draft model is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) techniques to minimize the Kullback-Leibler (KL) divergence between the draft distribution $P_D(x)$ and the target domain distribution.

* **Draft Model:** Llama-3.2-1B
* **Target Domain:** Python Code (flytech/python-codes-25k)
* **Technique:** QLoRA (Quantized Low-Rank Adaptation)

### 2.2 Online Speculative Inference
The inference pipeline implements the speculative sampling algorithm.
1.  **Drafting:** The draft model autoregressively generates a sequence of $\gamma$ tokens (lookahead window).
2.  **Verification:** The frozen target model (Llama-3.1-8B) computes probabilities for the candidate sequence in a single forward pass.
3.  **Rejection Sampling:** Tokens are accepted if $P_T(x) \geq P_D(x)$; otherwise, they are rejected based on a modified distribution to ensure the final output strictly adheres to the target model's probability distribution.

## 3. Methodology

### 3.1 Experimental Setup
All experiments are conducted on an NVIDIA T4 GPU (16GB VRAM) environment.
* **Baseline:** Standard autoregressive generation (Llama-3.1-8B).
* **Control:** Speculative decoding with a generic, pre-trained Llama-3.2-1B drafter.
* **Experiment:** Speculative decoding with the domain-tuned Llama-3.2-1B drafter.

### 3.2 Metrics
* **Token Acceptance Rate (TAR):** The mean ratio of accepted tokens per speculation step.
* **Speedup Factor:** $Latency_{baseline} / Latency_{speculative}$
* **Memory Overhead:** Additional VRAM consumption required for the draft model.

## 4. Repository Structure

```text
speculative-decoding-research/
│
├── README.md               # The formal research document
├── requirements.txt        # Dependencies (transformers, bitsandbytes, unsloth)
├── .gitignore              # Ignore large files (datasets, model weights, venv)
├── LICENSE                 # MIT License
│
├── data/                   # Store small raw datasets here
│   └── .gitkeep            # Keeps folder tracked even if empty
│
├── src/                    # Core modular logic (Python scripts)
│   ├── __init__.py
│   ├── benchmark.py        # Class to measure latency and VRAM
│   ├── spec_decoding.py    # Implementation of the speculation loop
│   └── utils.py            # Helpers for loading models/tokenizers
│
├── experiments/            # Jupyter Notebooks for analysis
│   ├── 01_baseline_analysis.ipynb   # Measuring generic draft performance
│   ├── 02_draft_finetuning.ipynb    # Unsloth training script
│   └── 03_comparative_results.ipynb # Final charts & speedup analysis
│
└── results/                # Generated plots and logs
    └── speedup_comparison.png
```
## 5. Preliminary Results
* *Note: This research is currently in the data collection phase.*
* Expected Outcome: Domain adaptation is projected to increase TAR by 15-20% on Python coding tasks.

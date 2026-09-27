# German Dialect Normalization with LLMs 🇩🇪🤖

This repository contains the evaluation codebase and datasets for an empirical NLP study on normalizing regional German dialects (Bavarian and Franconian) into Standard German using Large Language Models (LLMs) and In-Context Learning (ICL).

## Overview

We evaluate the capability of autoregressive LLMs (`qwen/qwen3.8-27b` via the Groq API) to perform dialect-to-standard-German translation. Our experimental design measures the impact of few-shot prompting compared to zero-shot extraction and a raw identity baseline. 

The evaluation leverages the **Betthupferl** dataset, encompassing 7 distinct dialect regions (e.g., Mittelfranken, Niederbayern, Oberbayern, etc.).

### Experimental Conditions
1. **Baseline-0 (Identity)**: Raw dialect evaluated directly against the Standard German reference without any translation.
2. **Condition-A (Zero-Shot)**: The LLM translates the dialect into Standard German with a single system instruction and no examples.
3. **Condition-B (Few-Shot)**: The LLM translates the dialect using 5 regional in-context demonstration pairs.

## Results Summary

Few-shot in-context learning dramatically improves translation fidelity over both the baseline and zero-shot approaches. The LLM-as-a-Judge module (deterministic scoring, Temperature = 0.0) confirms these improvements scale perfectly with traditional metrics (BLEU, WER, CER).

| Condition | BLEU (↑) | WER (↓) | CER (↓) | Meaning Pres. (1-5) (↑) | Fluency (1-5) (↑) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline-0 (Identity)** | 26.07 | 0.5804 | 0.2206 | 1.943 | 1.138 |
| **Condition-A (Zero-Shot)** | 49.97 | 0.2782 | 0.1892 | 3.752 | 4.586 |
| **Condition-B (Few-Shot)** | **59.76** | **0.1983** | **0.1337** | **4.010** | **4.600** |

All evaluation figures (Heatmaps, Radar Charts, WER comparisons) are automatically generated and saved in the `figures/` directory.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahul8020/German-Dialect-Normalization-with-LLM.git
   cd German-Dialect-Normalization-with-LLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

To run the complete evaluation pipeline (generation, traditional metrics, LLM-as-a-Judge, and visualizations):

```bash
python evaluate.py
```

*Note: The pipeline supports checkpointing. If interrupted by API rate limits, re-run with `python evaluate.py --load-checkpoints` to resume safely from the last saved state without regenerating existing data.*

## Citations & Acknowledgments

This project utilizes the multi-dialectal dataset for German dialects presented in:
> Verena Blaschke, Miriam Winkler, Constantin Förster, Gabriele Wenger-Glemser, and Barbara Plank. "A multi-dialectal dataset for German dialect ASR and dialect-to-standard speech translation." *In Proc. Interspeech 2025*, p. 913–917. ISCA. [https://www.isca-archive.org/interspeech_2025/blaschke25_interspeech.html](https://www.isca-archive.org/interspeech_2025/blaschke25_interspeech.html)

If you use the data or codebase from this repository, please cite:
```bibtex
@inproceedings{blaschke-etal-2025-multi,
  title = {A Multi-Dialectal Dataset for {German} Dialect {ASR} and Dialect-to-Standard Speech Translation},
  author = {Blaschke, Verena and Winkler, Miriam and Förster, Constantin and Wenger-Glemser, Gabriele and Plank, Barbara},
  booktitle = {Interspeech 2025},
  pages = {913--917},
  url = {https://www.isca-archive.org/interspeech_2025/blaschke25_interspeech.html},
  doi = {10.21437/Interspeech.2025-318},
  year = {2025},
  month = aug,
}
```

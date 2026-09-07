# CaseSumm

Code and saved experiment results for **CaseSumm: A Large-Scale Dataset for Long-Context Summarization from U.S. Supreme Court Opinions** (Findings of NAACL 2025).

[Paper](https://aclanthology.org/2025.findings-naacl.102/) · [Dataset](https://huggingface.co/datasets/ChicagoHAI/CaseSumm)

CaseSumm pairs Supreme Court opinions with official syllabuses. We study long-document summarization and compare automatic evaluation with law-student judgments.

## Setup

Use Python 3.12 and run commands from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
```

For API generation, add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to `.env`. Existing environment variables take precedence. Local data, outputs, caches and checkpoints are ignored by Git.

Local Mistral inference needs a compatible vLLM/CUDA installation.

## Data and saved results

Download the opinion–syllabus corpus from the dataset link above. The [artifacts](artifacts/) directory contains about 21 MB of recovered results:

- Summaries from six models for 622 cases, with separate versions used in human evaluation and G-Eval
- 285 anonymous ratings from 57 assessments of 33 cases
- Saved train/dev/test case records, citation corrections and automatic scores
- G-Eval prompts, scores and source checksums

Model weights, full opinions, participant metadata and third-party reference summary text are not included.

## Usage

Generation and evaluation accept local JSON, JSONL, CSV or Hugging Face `save_to_disk` directories. Generation needs `citation` and `opinion`. Evaluation also needs `generated` and `syllabus`. Use `--split` for a saved DatasetDict or records with a `split` field.

Preview two prompts without calling an API:

```sh
python -m src.generation.llm_summarize \
  --input data/cases.jsonl --sample 2 \
  --output outputs/prompt_preview.json --dry-run
```

Remove `--dry-run` and choose a new output path to generate summaries. The default GPT model is `gpt-4-1106-preview`. Use `--model-id` to select another model. Changing the model produces a new experiment.

Score predictions with ROUGE:

```sh
python -m src.evaluation.auto_metrics \
  --input outputs/predictions.json --output outputs/metrics.json
```

ROUGE uses stemming and reports per-case scores and macro means. Add `--bertscore-model CHECKPOINT` for BERTScore. BARTScore and G-Eval are included as saved results, not live scoring commands.

Other entry points support `--help`:

| Task | Module |
| --- | --- |
| Collect and prepare opinions | `src.preprocessing.collect`, `src.preprocessing.build_dataset` |
| Optimize prompts | `src.generation.dspy_prompt` |
| Fine-tune Mistral | `src.training.sft_train` |
| Merge an adapter | `src.training.merge_adapter` |
| Run local inference | `src.generation.predict` |
| Convert flattened study exports | `src.human_eval.process_labelstudio_annotations` |

## Tests

Run the offline tests:

```sh
python -m unittest discover -s tests -v
```

These checks need no API credentials or pretrained weights.

## Citation

Code by Mourad Heddaya and Kyle MacMillan. The fine-tuning code adapts [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

```bibtex
@inproceedings{heddaya-etal-2025-casesumm,
  title = {CaseSumm: A Large-Scale Dataset for Long-Context Summarization from U.S. Supreme Court Opinions},
  author = {Heddaya, Mourad and MacMillan, Kyle and Mei, Hongyuan and Tan, Chenhao and Malani, Anup},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2025},
  year = {2025},
  url = {https://aclanthology.org/2025.findings-naacl.102/}
}
```

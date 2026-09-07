"""Generate case summaries with GPT or Claude."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import DATA_DIR
from src.data_io import load_records, require_fields, write_json

if TYPE_CHECKING:
    from src.generation.cache import APICache

MODEL_IDS = {
    "gpt4": "gpt-4",
    "gpt4t": "gpt-4-1106-preview",
    "gpt35": "gpt-3.5-turbo-1106",
    "claude": "claude-3-opus-20240229",
}
PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "gpt"


def render_prompt(row: dict[str, Any], template: str) -> str:
    """Fill a summary prompt with the case text.

    template: Prompt text containing {OPINION} and optionally {NUM_PARAGRAPHS}.
    row: Case record. The syllabus_len field is also required for {NUM_PARAGRAPHS}.
    """
    paragraph_count = 0
    if "{NUM_PARAGRAPHS}" in template:
        if "syllabus_len" not in row:
            raise ValueError(
                "The simple prompt uses syllabus_len to set the summary length. Supply that field or use the optimized prompt"
            )
        paragraph_count = int(float(row["syllabus_len"]) / 150)
    return template.format(
        OPINION=row["opinion"], NUM_PARAGRAPHS=paragraph_count, HUMAN_PROMPT="", AI_PROMPT=""
    )


def generate_records(
    rows: list[dict[str, Any]],
    model: str,
    template: str,
    cache: "APICache",
    max_tokens: int = 1000,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return case records with generated summaries and API response details.

    rows: Case records containing citation and opinion.
    model: Alias from MODEL_IDS, also used to select provider-specific handling.
    model_id: Override the API model ID associated with the alias.
    max_tokens: Maximum number of output tokens per summary.
    """
    require_fields(rows, ["citation", "opinion"])
    prompts = [render_prompt(row, template) for row in rows]
    model_id = model_id or MODEL_IDS[model]
    results = []
    for row, prompt in zip(rows, prompts):
        if model == "gpt35":
            import tiktoken

            encoding = tiktoken.encoding_for_model(model_id)
            prompt = encoding.decode(encoding.encode(prompt)[: 16378 - max_tokens])
        response = cache.generate(
            model=model_id,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        if model == "claude":
            summary = "".join(
                block["text"] for block in response["content"] if block["type"] == "text"
            )
            tokens = response["usage"]["output_tokens"]
            finish = response["stop_reason"]
        else:
            summary = response["choices"][0]["message"].get("content")
            tokens = response["usage"]["completion_tokens"]
            finish = response["choices"][0]["finish_reason"]
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"Provider returned no summary for {row['citation']}")
        result = dict(row)
        result.update(
            generated=summary.strip(),
            model=model_id,
            completion_tokens=tokens,
            finish_reason=finish,
        )
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Saved HF dataset, JSON, JSONL or CSV"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--model", choices=MODEL_IDS, default="gpt4t")
    parser.add_argument("--model-id", help="Model ID to use instead of the selected model alias")
    parser.add_argument("--prompt", choices=["simple", "optimized"], default="optimized")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a prompt template, required for Claude",
    )
    parser.add_argument("--sample", type=int, help="First N records in input order")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cache" / "generations.sqlite3")
    parser.add_argument(
        "--dry-run", action="store_true", help="Write prepared prompts without a provider"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Choose a new path or pass --overwrite")
    if args.max_tokens < 1 or (args.model == "gpt35" and args.max_tokens >= 16378):
        parser.error("Invalid output token budget")
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be positive")
    if args.model == "claude" and args.prompt_file is None:
        parser.error("Claude requires --prompt-file. The included templates are for GPT")
    template = (args.prompt_file or PROMPT_DIR / f"summarize_{args.prompt}.txt").read_text()
    rows = load_records(args.input, args.split)
    if args.sample is not None:
        rows = rows[: args.sample]
    require_fields(rows, ["citation", "opinion"])
    prompts = [render_prompt(row, template) for row in rows]
    if args.dry_run:
        results = [
            {
                "citation": row["citation"],
                "model": args.model_id or MODEL_IDS[args.model],
                "prompt": prompt,
            }
            for row, prompt in zip(rows, prompts)
        ]
    else:
        from src.generation.cache import ClaudeAPICache, OpenAIAPICache

        cache_type = ClaudeAPICache if args.model == "claude" else OpenAIAPICache
        cache = cache_type(cache_path=args.cache)
        results = generate_records(
            rows, args.model, template, cache, args.max_tokens, args.model_id
        )
    write_json(args.output, results, overwrite=args.overwrite)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()

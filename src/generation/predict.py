"""Generate case summaries with Mistral using vLLM."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data_io import load_records, require_fields, write_json
from src.training.prompts import PROMPT_DICT

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM, SamplingParams

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"


def tokenize_prompts(
    rows: list[dict[str, Any]],
    tokenizer: "PreTrainedTokenizerBase",
    context_length: int = 32768,
    max_tokens: int = 1500,
) -> list[list[int]]:
    """Tokenize prompts and truncate opinions to leave room for the summary.

    context_length: Total token limit for the prompt and generated summary.
    max_tokens: Tokens reserved for the summary within that limit.
    """
    require_fields(rows, ["citation", "opinion"])
    closing_tokens = tokenizer.encode("\n\n[TEXT_END]\n\n[/INST]", add_special_tokens=False)
    prompt_budget = context_length - max_tokens
    if prompt_budget <= len(closing_tokens):
        raise ValueError("Context length must leave room for the prompt and output")
    prompts = []
    for row in rows:
        # The archived template already includes <s>; do not prepend another BOS token.
        tokens = tokenizer.encode(
            PROMPT_DICT["prompt_infer"].format(opinion=row["opinion"]), add_special_tokens=False
        )
        if len(tokens) > prompt_budget:
            tokens = tokens[: prompt_budget - len(closing_tokens)] + closing_tokens
        prompts.append(tokens)
    return prompts


def generate_with_retries(
    engine: "LLM",
    prompts: list[list[int]],
    parameters: "SamplingParams",
    backup_parameters: "SamplingParams | None" = None,
    max_retries: int = 4,
) -> list[dict[str, Any]]:
    """Generate summaries, retrying any that reach the token limit.

    prompts: One list of token IDs per case.
    parameters: vLLM sampling settings for the first attempt.
    backup_parameters: Sampling settings for retries. None disables retries.
    max_retries: Maximum additional attempts per case.
    """
    outputs = engine.generate(prompt_token_ids=prompts, sampling_params=parameters)
    if len(outputs) != len(prompts):
        raise ValueError("Inference engine returned a different number of records")
    results = []
    for index, output in enumerate(outputs):
        completion = output.outputs[0]
        retries = 0
        while (
            backup_parameters is not None
            and completion.finish_reason == "length"
            and retries < max_retries
        ):
            # Retry the same case and keep its position in the results.
            retry = engine.generate(
                prompt_token_ids=[prompts[index]], sampling_params=backup_parameters
            )
            if len(retry) != 1:
                raise ValueError("Expected exactly one regenerated summary")
            completion = retry[0].outputs[0]
            retries += 1
        results.append(
            {
                "generated": completion.text.strip(),
                "finish_reason": completion.finish_reason,
                "regeneration_attempts": retries,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default=MODEL_ID,
        help="Base model ID or already-merged fine-tuned model directory",
    )
    parser.add_argument("--sample", type=int)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate-length-finish", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Export text prompts without loading vLLM or a model"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Choose a new path or pass --overwrite")
    if (
        args.max_tokens < 1
        or args.context_length <= args.max_tokens
        or (args.sample is not None and args.sample < 1)
    ):
        parser.error("Invalid token budget or sample size")
    rows = load_records(args.input, args.split)
    if args.sample is not None:
        rows = rows[: args.sample]
    require_fields(rows, ["citation", "opinion"])
    if args.dry_run:
        results = [
            dict(
                row,
                model=args.model_path,
                prompt=PROMPT_DICT["prompt_infer"].format(opinion=row["opinion"]),
            )
            for row in rows
        ]
    else:
        from vllm import LLM, SamplingParams

        engine = LLM(model=args.model_path, seed=args.seed, max_model_len=args.context_length)
        prompts = tokenize_prompts(
            rows, engine.get_tokenizer(), args.context_length, args.max_tokens
        )
        sampling_params = SamplingParams(
            temperature=0,
            repetition_penalty=1.25,
            stop=[],
            max_tokens=args.max_tokens,
            include_stop_str_in_output=False,
        )
        retry_params = (
            SamplingParams(
                temperature=1,
                top_p=0.9,
                repetition_penalty=1.3,
                stop=[],
                max_tokens=args.max_tokens,
                include_stop_str_in_output=False,
            )
            if args.regenerate_length_finish
            else None
        )
        generated = generate_with_retries(engine, prompts, sampling_params, retry_params)
        results = [
            dict(row, **result, model=args.model_path, seed=args.seed)
            for row, result in zip(rows, generated)
        ]
    write_json(args.output, results, overwrite=args.overwrite)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()

"""Optimize summarization prompts with DSPy."""

import argparse
import os
from collections.abc import Callable
from contextlib import chdir
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import DATA_DIR, require_secret
from src.data_io import load_records, require_fields, write_json

if TYPE_CHECKING:
    import dspy

DEFAULT_MODEL = "gpt-4-1106-preview"
os.environ.setdefault("DSP_CACHEDIR", str(DATA_DIR / "cache" / "dspy"))
os.environ.setdefault("DSP_NOTEBOOK_CACHEDIR", str(DATA_DIR / "cache" / "dspy-notebook"))


def score_rouge(
    gold: "dict[str, Any] | dspy.Example",
    pred: "dict[str, Any] | dspy.Prediction",
    trace: Any = None,
) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    return scorer.score(gold["syllabus"], pred["syllabus"])["rouge2"].fmeasure


def build_summarizer(optimized: bool = False) -> "dspy.Module":
    """Create a DSPy summarizer from one of the prompt templates.

    optimized: Select the longer instruction template. This does not run
        prompt optimization.
    """
    import dspy

    class SummarizerSignature(dspy.Signature):
        """Summarize the Supreme Court Opinion."""

        opinion = dspy.InputField(desc="SCOTUS Opinion")
        syllabus = dspy.OutputField(desc="SCOTUS Syllabus")

    class SummarizerSignatureOptimized(dspy.Signature):
        """Review the provided Supreme Court opinion text. Deliver a concise, neutral summary that captures the essence of the legal reasoning, main points of law, conclusions drawn, and the implications of the decision, all whilst adhering to comprehensible language suitable for an educated general audience."""

        opinion = dspy.InputField(desc="SCOTUS Opinion", prefix="Opinion:")
        syllabus = dspy.OutputField(
            desc="SCOTUS Syllabus", prefix="Summary of Supreme Court Opinion:"
        )

    class NoCoT(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generate_answer = dspy.Predict(
                SummarizerSignatureOptimized if optimized else SummarizerSignature
            )

        def forward(self, opinion: str) -> "dspy.Prediction":
            return self.generate_answer(opinion=opinion)

    return NoCoT()


def select_examples(rows: list[dict[str, Any]], sample: int, seed: int) -> list[dict[str, Any]]:
    """Sample cases using a seeded random shuffle.

    sample: Maximum number of cases to select. Returns all if fewer are available.
    """
    require_fields(rows, ["citation", "opinion", "syllabus"])
    rows = list(rows)
    import numpy as np

    # Use the same random shuffle as Hugging Face Dataset.shuffle.
    indices = np.random.default_rng(seed).permutation(len(rows))[:sample]
    return [rows[index] for index in indices]


def validate_disjoint(optimization: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> None:
    shared = {row["citation"] for row in optimization} & {row["citation"] for row in evaluation}
    if shared:
        raise ValueError(f"Optimization and evaluation share {len(shared)} case identifiers")


def run_dspy(
    optimization: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    num_threads: int = 1,
    lm: Any = None,
    optimizer_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], float]:
    """Optimize the prompt, then evaluate it on a separate set of cases.

    optimization, evaluation: Case records with citation, opinion and syllabus.
        The two sets must not contain any of the same cases.
    lm: Optional DSPy language model to use instead of creating one from model.
    optimizer_factory: Optional optimizer class or callable. Defaults to
        SignatureOptimizer.

    Returns the program state and evaluation ROUGE-2 F1 as a percentage.
    """
    import dspy
    from dspy.evaluate import Evaluate
    from dspy.teleprompt import SignatureOptimizer

    require_fields(optimization, ["citation", "opinion", "syllabus"])
    require_fields(evaluation, ["citation", "opinion", "syllabus"])
    validate_disjoint(optimization, evaluation)
    optimization = [dspy.Example(**row).with_inputs("opinion") for row in optimization]
    evaluation = [dspy.Example(**row).with_inputs("opinion") for row in evaluation]
    if lm is None:
        lm = dspy.OpenAI(
            model=model,
            api_key=require_secret("OPENAI_API_KEY"),
            max_tokens=1000,
            model_type="chat",
        )
    # Use this language model only for the duration of the run.
    with dspy.context(lm=lm):
        optimizer = (optimizer_factory or SignatureOptimizer)(metric=score_rouge, verbose=False)
        compiled = optimizer.compile(
            build_summarizer(),
            devset=optimization,
            eval_kwargs={
                "num_threads": num_threads,
                "display_progress": False,
                "display_table": False,
            },
        )
        evaluator = Evaluate(
            devset=evaluation,
            metric=score_rouge,
            num_threads=num_threads,
            display_progress=False,
            display_table=False,
        )
        score = evaluator(compiled)
    return compiled.dump_state(), float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Saved HF DatasetDict or split-labeled records"
    )
    parser.add_argument(
        "--optimization-split",
        choices=["train", "dev"],
        default="train",
        help="train follows the paper. Dev follows the original script call",
    )
    parser.add_argument("--evaluation-split", choices=["dev", "test"], default="test")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sample < 1 or args.num_threads < 1:
        parser.error("--sample and --num-threads must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Choose a new path or pass --overwrite")
    optimization = select_examples(
        load_records(args.input, args.optimization_split), args.sample, args.seed
    )
    evaluation = select_examples(
        load_records(args.input, args.evaluation_split), args.sample, args.seed
    )
    validate_disjoint(optimization, evaluation)
    result = {
        "model": args.model_id,
        "optimizer": "SignatureOptimizer",
        "dspy_version": version("dspy-ai"),
        "optimization_split": args.optimization_split,
        "evaluation_split": args.evaluation_split,
        "seed": args.seed,
        "optimization_citations": [r["citation"] for r in optimization],
        "evaluation_citations": [r["citation"] for r in evaluation],
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        # DSPy writes logs during import, so run it from the cache directory.
        runtime = DATA_DIR / "cache" / "dspy-runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with chdir(runtime):
            state, score = run_dspy(optimization, evaluation, args.model_id, args.num_threads)
        result.update(program_state=state, evaluation_rouge2_percent=score)
    write_json(args.output, result, overwrite=args.overwrite)
    print(f"Saved DSPy run to {args.output}")


if __name__ == "__main__":
    main()

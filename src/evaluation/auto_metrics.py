"""Evaluate generated summaries with ROUGE and optional BERTScore."""

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from src.data_io import load_records, require_fields, write_json


def score_records(
    rows: list[dict[str, Any]],
    prediction_column: str = "generated",
    reference_column: str = "syllabus",
    bert_model: str | None = None,
) -> dict[str, Any]:
    """Return scores for each case and the average across cases.

    prediction_column, reference_column: Record fields containing the generated
        and reference summaries.
    bert_model: BERTScore model checkpoint. None skips BERTScore.

    Empty summaries receive zero ROUGE scores.
    """
    from rouge_score import rouge_scorer

    require_fields(rows, ["citation", reference_column])
    for index, row in enumerate(rows):
        if not isinstance(row.get(prediction_column), str):
            raise ValueError(
                f"Record {index}: {prediction_column} must be a string (empty is scored as zero)"
            )
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []
    for row in rows:
        scores = scorer.score(row[reference_column], row[prediction_column])
        result = {"citation": row["citation"]}
        for name, score in scores.items():
            result.update(
                {
                    f"{name}_precision": score.precision,
                    f"{name}_recall": score.recall,
                    f"{name}_f1": score.fmeasure,
                }
            )
        results.append(result)
    if bert_model is not None:
        from bert_score import score

        precision, recall, f1, metric_hash = score(
            [row[prediction_column] for row in rows],
            [row[reference_column] for row in rows],
            model_type=bert_model,
            lang="en",
            rescale_with_baseline=False,
            return_hash=True,
        )
        for result, p, r, f in zip(results, precision.tolist(), recall.tolist(), f1.tolist()):
            result.update(bertscore_precision=p, bertscore_recall=r, bertscore_f1=f)
    metrics = [key for key in results[0] if key != "citation"]
    report = {
        "count": len(rows),
        "prediction_column": prediction_column,
        "reference_column": reference_column,
        "rouge_use_stemmer": True,
        "aggregation": "macro mean",
        "rouge_scale": "[0,1]",
        "aggregate": {key: mean(row[key] for row in results) for key in metrics},
        "per_case": results,
    }
    if bert_model is not None:
        report.update(
            bertscore_model=bert_model,
            bertscore_hash=metric_hash,
            bertscore_rescale_with_baseline=False,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-column", default="generated")
    parser.add_argument("--reference-column", default="syllabus")
    parser.add_argument(
        "--bertscore-model",
        help="Model checkpoint for BERTScore, requires bert-score and torch",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Choose a new path or pass --overwrite")
    report = score_records(
        load_records(args.input, args.split),
        args.prediction_column,
        args.reference_column,
        args.bertscore_model,
    )
    write_json(args.output, report, overwrite=args.overwrite)
    print(f"Scored {report['count']} summaries. Report: {args.output}")


if __name__ == "__main__":
    main()

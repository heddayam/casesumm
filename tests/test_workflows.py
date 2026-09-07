"""Offline checks for case alignment, preprocessing, generation and evaluation."""

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.data_io import load_records, write_json
from src.evaluation.auto_metrics import score_records
from src.generation.dspy_prompt import (
    run_dspy,
    validate_disjoint,
)
from src.generation.predict import generate_with_retries, tokenize_prompts

ROOT = Path(__file__).resolve().parents[1]
BUILDER = importlib.import_module("src.preprocessing.build_dataset")


def record(citation="synthetic-1"):
    return {
        "citation": citation,
        "opinion": "The court reversed the judgment and remanded for a new hearing. " * 8,
        "syllabus": "Held: the judgment was reversed.",
        "opinion_len": 100,
        "syllabus_len": 6,
        "structured_syllabus": True,
    }


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        previous_directory = Path.cwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, previous_directory)
        environment = patch.dict(
            os.environ,
            {
                "DSP_CACHEDIR": str(self.work / "dspy"),
                "DSP_NOTEBOOK_CACHEDIR": str(self.work / "dspy-notebook"),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def test_saved_dataset_split_roundtrip(self):
        from datasets import Dataset, DatasetDict

        rows = [record(str(i)) for i in range(3)]
        splits = BUILDER.make_structured_splits(rows, {"1"}, {"2"})
        dataset = DatasetDict(
            {name: Dataset.from_list(records) for name, records in splits.items()}
        )
        dataset.save_to_disk(str(self.work / "dataset"))
        self.assertEqual(load_records(self.work / "dataset", "dev")[0]["citation"], "1")
        with self.assertRaises(ValueError):
            load_records(self.work / "dataset")
        with self.assertRaises(ValueError):
            BUILDER.make_structured_splits(rows, {"1"}, {"1"})
        with self.assertRaises(ValueError):
            BUILDER.make_structured_splits(rows, {"missing"}, {"2"})

    def test_corpus_builder_filters_and_rejects_duplicate_cases(self):
        encoder = Mock()
        encoder.encode.side_effect = lambda text: text.split()
        raw = self.work / "super.jsonl"
        raw.write_text(
            json.dumps(record())
            + "\n"
            + json.dumps({"citation": "empty", "opinion": "", "syllabus": "Held: x"})
            + "\n"
        )
        rows = BUILDER.clean_records(BUILDER.collect_records(encoder, super_scotus=raw))
        self.assertEqual([r["citation"] for r in rows], ["synthetic-1"])
        with self.assertRaises(ValueError):
            BUILDER.clean_records(rows + rows)

    def test_super_scotus_preprocessing_preserves_case_alignment(self):
        module = importlib.import_module("src.preprocessing.process_super_scotus")
        row = {
            "citation": "600 US _",
            "id": "15",
            "justia_sections": {
                "Opinion": "Justice A delivered the opinion of the Court.\nThe court reversed.",
                "Syllabus": "Decided June 1, 2023\nHeld: the judgment was reversed.",
            },
        }
        rows = module.preprocess_records([row, {"citation": "601 US 2", "justia_sections": {}}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["citation"], "600.US.15")
        self.assertEqual(rows[0]["opinion"], "The court reversed.")
        self.assertEqual(rows[0]["syllabus"], "Held: the judgment was reversed.")
        with self.assertRaises(ValueError):
            module.preprocess_records([{"citation": "bad"}])

    def test_rouge_identical_and_empty_predictions(self):
        rows = [dict(record("a"), generated=record()["syllabus"]), dict(record("b"), generated="")]
        report = score_records(rows)
        self.assertEqual(report["per_case"][0]["rouge2_f1"], 1)
        self.assertEqual(report["per_case"][1]["rouge2_f1"], 0)
        self.assertEqual(report["aggregate"]["rouge2_f1"], 0.5)
        write_json(self.work / "metrics.json", report)

    def test_real_signature_optimizer_compiles_with_offline_lm(self):
        from dspy.teleprompt import SignatureOptimizer
        from dspy.utils.dummies import DummyLM

        lm = DummyLM(
            {
                "You are an instruction optimizer": "Summarize carefully.\nProposed Prefix For Output Field: Syllabus:",
                "": record()["syllabus"],
            }
        )
        with contextlib.redirect_stdout(io.StringIO()):
            state, score = run_dspy(
                [record("train")],
                [record("test")],
                lm=lm,
                optimizer_factory=partial(SignatureOptimizer, breadth=2, depth=1),
            )
        self.assertEqual(score, 100)
        write_json(self.work / "program.json", state)
        with self.assertRaises(ValueError):
            validate_disjoint([record()], [record()])

    def test_regeneration_preserves_case_alignment_and_bounds_retries(self):
        def output(text, reason):
            return SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason=reason)])

        engine = Mock()
        engine.generate.side_effect = [
            [output("first", "stop"), output("second truncated", "length")],
            [output("second still truncated", "length")],
            [output("second final", "stop")],
        ]
        rows = generate_with_retries(engine, [[1], [2]], "primary", "backup", max_retries=2)
        self.assertEqual([row["generated"] for row in rows], ["first", "second final"])
        self.assertEqual(engine.generate.call_args_list[1].kwargs["prompt_token_ids"], [[2]])
        self.assertEqual(rows[1]["regeneration_attempts"], 2)

    def test_token_budget_keeps_closing_instruction(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[9, 9], list(range(30))]
        self.assertEqual(
            tokenize_prompts([record()], tokenizer, context_length=10, max_tokens=3),
            [[0, 1, 2, 3, 4, 9, 9]],
        )

    def test_annotation_mapping_missingness_and_private_field_exclusion(self):
        module = importlib.import_module("src.human_eval.process_labelstudio_annotations")
        data = {
            "citation": "synthetic",
            "summary_order": "mistral,gpt4,syllabus,oyez,westlaw",
            "email": "private-placeholder",
            "summary0": "Do not export this text",
            "error1": "Yes",
            "error2": "No",
        }
        for field in module.DIMENSIONS:
            data.update({f"{field}{rank}": letter for rank, letter in enumerate("EDCBA", 1)})
        rows = module.convert_assessment(data, "rater_001", "assessment_0001")
        self.assertEqual(rows[0]["model"], "mistral")
        self.assertEqual(rows[0]["sensitivity"], 5)
        self.assertTrue(rows[0]["factual_error"])
        self.assertIsNone(rows[2]["factual_error"])
        self.assertFalse(any("email" in row or "summary0" in row for row in rows))
        data["relevant1"] = "A"
        with self.assertRaises(ValueError):
            module.convert_assessment(data, "rater_001", "assessment_0001")

    def test_html_converters_handle_missing_footer_and_final_paragraph(self):
        older = importlib.import_module("src.preprocessing.pro_to_json")
        alternative = importlib.import_module("src.preprocessing.pro_to_json_sections")
        html = '<title>Synthetic case</title><div class="num">\n1\nOnly paragraph survives.</div>'
        self.assertIn("Only paragraph survives.", older.process_html(html)["majority"][0])
        self.assertIn(
            "Only paragraph survives.",
            str(alternative.Case(html, "100.US.1.html").data["majority"]),
        )

    def test_extraction_empty_pdf_and_in_memory_footer_detection(self):
        extractor = importlib.import_module("src.preprocessing.extract_syllabus")
        line_detect = importlib.import_module("src.preprocessing.line_detect")
        from PIL import Image, ImageDraw

        raster = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(raster).line((10, 100, 180, 100), fill="black", width=2)
        page = Mock()
        page.to_image.return_value.original = raster
        self.assertAlmostEqual(line_detect.detect_footer_line(page), 48, delta=2)
        page.extract_text.return_value = None
        self.assertEqual(
            extractor.extract_syllabus(SimpleNamespace(pages=[page]), 100, "Synthetic"), ("", "")
        )
        with self.assertRaises(ValueError):
            extractor.extract_syllabus(SimpleNamespace(pages=[]), 100, "Synthetic")


class DataProcessingTests(unittest.TestCase):
    def test_html_converter_separates_paragraphs_and_switches_sections(self):
        from src.preprocessing.pro_to_json import process_html

        paragraphs = [
            "First sentence.",
            "Second sentence.",
            "Justice A, dissenting.",
            "Dissent body.",
            "Justice B, concurring.",
            "Concurrence body.",
        ]
        html = "".join(f'<div class="num">\n{i}\n{text}</div>' for i, text in enumerate(paragraphs))
        result = process_html(html)
        self.assertEqual(result["majority"], ["First sentence.\nSecond sentence."])
        self.assertIn("Dissent body.", result["dissent"][0])
        self.assertNotIn("Concurrence body.", result["dissent"][0])
        self.assertIn("Concurrence body.", result["concurrence"][0])

    def test_metrics_cli_selects_requested_split(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            rows = [
                dict(record(name), split=name, generated=record()["syllabus"])
                for name in ("train", "test")
            ]
            write_json(work / "input.json", rows)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.evaluation.auto_metrics",
                    "--input",
                    str(work / "input.json"),
                    "--output",
                    str(work / "metrics.json"),
                    "--split",
                    "test",
                ],
                cwd=work,
                env=dict(os.environ, PYTHONPATH=str(ROOT)),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((work / "metrics.json").read_text())
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["per_case"][0]["citation"], "test")

    def test_label_studio_controls_target_text_objects(self):
        import xml.etree.ElementTree as ET

        tree = ET.parse(ROOT / "src/human_eval/label_studio_interface.html")
        objects = {node.attrib["name"] for node in tree.iter() if node.tag in {"Text", "HyperText"}}
        controls = {node.attrib["name"]: node for node in tree.iter("Choices")}
        for node in controls.values():
            self.assertIn(node.attrib["toName"], objects)
        for dimension in ("relevant", "irrelevant", "clear", "style"):
            for rank in range(1, 6):
                self.assertEqual(
                    {node.attrib["value"] for node in controls[f"{dimension}{rank}"]}, set("ABCDE")
                )


if __name__ == "__main__":
    unittest.main()

"""Exercise the real training entry point with a tiny offline Mistral model."""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrainingSmokeTest(unittest.TestCase):
    def test_sentencepiece_tokenizer_loads_for_mistral_training(self):
        import sentencepiece as spm
        from transformers import AutoTokenizer, LlamaTokenizer

        from src.generation.predict import tokenize_prompts
        from src.training.prompts import PROMPT_DICT

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            corpus = work / "corpus.txt"
            corpus.write_text(
                PROMPT_DICT["prompt"].format(
                    opinion="The court reviewed the case and reversed the judgment.",
                    syllabus="Held: the judgment was reversed.",
                ),
                encoding="utf-8",
            )
            spm.SentencePieceTrainer.train(
                input=str(corpus),
                model_prefix=str(work / "tokenizer"),
                vocab_size=64,
                hard_vocab_limit=False,
                minloglevel=2,
            )
            tokenizer = LlamaTokenizer(vocab_file=str(work / "tokenizer.model"))
            tokenizer.save_pretrained(work / "saved")
            loaded = AutoTokenizer.from_pretrained(
                work / "saved", use_fast=False, local_files_only=True
            )
            self.assertIsInstance(loaded, LlamaTokenizer)
            tokens = tokenize_prompts(
                [{"citation": "synthetic", "opinion": "The court reviewed the case."}],
                loaded,
                context_length=512,
                max_tokens=32,
            )[0]
            self.assertEqual(tokens[0], loaded.bos_token_id)
            self.assertEqual(tokens.count(loaded.bos_token_id), 1)

    def test_one_step_offline_training_and_adapter_merge(self):
        from datasets import Dataset, DatasetDict
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

        from src.training.prompts import PROMPT_DICT

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="casesumm training ") as folder:
            work = Path(folder)
            row = {
                "citation": "synthetic",
                "opinion": "The court reviewed a synthetic dispute and reversed the judgment. " * 5,
                "syllabus": "Held: the judgment was reversed.",
                "opinion_len": 80,
                "syllabus_len": 8,
                "structured_syllabus": True,
            }
            prompt = PROMPT_DICT["prompt"].format(**row)
            backend = Tokenizer(models.WordLevel(unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            backend.train_from_iterator(
                [prompt],
                trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "<s>", "</s>"]),
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="[UNK]",
                pad_token="[PAD]",
                bos_token="<s>",
                eos_token="</s>",
            )
            model_dir = work / "base"
            tokenizer.save_pretrained(model_dir)
            model = MistralForCausalLM(
                MistralConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=256,
                    sliding_window=128,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
            model.save_pretrained(model_dir)
            dataset_dir = work / "dataset"
            DatasetDict(
                {
                    name: Dataset.from_list([dict(row, citation=name)])
                    for name in ["train", "dev", "test"]
                }
            ).save_to_disk(str(dataset_dir))
            env = dict(
                os.environ,
                PYTHONPATH=str(root),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                MPLCONFIGDIR=str(work / "matplotlib"),
                TOKENIZERS_PARALLELISM="false",
            )
            output = work / "adapter"
            command = [
                sys.executable,
                "-m",
                "src.training.sft_train",
                "--model_name_or_path",
                str(model_dir),
                "--data_path",
                str(dataset_dir),
                "--output_dir",
                str(output),
                "--max_steps",
                "1",
                "--per_device_train_batch_size",
                "1",
                "--model_max_length",
                "256",
                "--use_cpu",
                "True",
                "--bf16",
                "False",
                "--report_to",
                "none",
                "--save_strategy",
                "no",
                "--logging_strategy",
                "no",
            ]
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=90
            )
            self.assertEqual(result.returncode, 0, result.stderr[-6000:])
            self.assertTrue((output / "adapter_config.json").is_file())
            state = json.loads((output / "trainer_state.json").read_text())
            self.assertEqual(state["global_step"], 1)
            loss = state["log_history"][-1]["train_loss"]
            self.assertTrue(math.isfinite(loss))
            self.assertGreater(loss, 0)
            from safetensors.torch import load_file

            adapter = load_file(output / "adapter_model.safetensors")
            self.assertTrue(
                any(
                    weights.count_nonzero().item() > 0
                    for name, weights in adapter.items()
                    if "lora_B" in name
                )
            )
            saved_adapter = (output / "adapter_model.safetensors").read_bytes()
            repeated = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Output is not empty", repeated.stderr)
            self.assertEqual((output / "adapter_model.safetensors").read_bytes(), saved_adapter)
            merged = work / "merged"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.training.merge_adapter",
                    "--adapter",
                    str(output),
                    "--output",
                    str(merged),
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            self.assertTrue((merged / "model.safetensors").is_file())
            self.assertTrue((merged / "tokenizer_config.json").is_file())


if __name__ == "__main__":
    unittest.main()

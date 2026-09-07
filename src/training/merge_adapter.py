"""Merge a saved LoRA adapter into its base model for vLLM inference."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists. Choose a new directory")
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = PeftConfig.from_pretrained(str(args.adapter))
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype="auto")
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    tokenizer_source = (
        args.adapter
        if (args.adapter / "tokenizer_config.json").exists()
        else config.base_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved merged model and tokenizer to {args.output}")


if __name__ == "__main__":
    main()

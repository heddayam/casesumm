# Modified for CaseSumm: Mistral LoRA fine-tuning, dataset filtering and portable configuration.
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from src.preprocessing.dataset import load_hf_dataset

os.environ.setdefault("WANDB_PROJECT", "scotus_summarization_sft")

from src.training.prompts import INSTR_END, PROMPT_DICT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.1")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=10000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    run_name: str
    report_to: str = field(default="none")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict[str, str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
) -> None:
    """Add special tokens and initialize their embeddings from the existing mean."""

    tokenizer.pad_token = tokenizer.unk_token

    if len(special_tokens_dict) != 0:
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizerBase, data_args: DataArguments
) -> dict[str, Any]:
    """Prepare training and dev splits with loss computed only on the summary."""
    logging.warning("Loading data...")
    dataset = load_hf_dataset(path=data_args.data_path, structured=True)

    logging.warning("Formatting inputs...")
    prompt = PROMPT_DICT["prompt"]

    dataset = dataset.filter(lambda x: x["opinion_len"] + x["syllabus_len"] + 20 < 8096)
    dataset = dataset.map(
        lambda x: {"prompt": prompt.format(opinion=x["opinion"], syllabus=x["syllabus"])}
    )

    template_ids = tokenizer.encode(INSTR_END, return_tensors="pt", add_special_tokens=False)[0][
        2:
    ].tolist()
    data_collator = DataCollatorForCompletionOnlyLM(template_ids, tokenizer=tokenizer, mlm=False)

    return dict(
        train_dataset=dataset["train"], eval_dataset=dataset["dev"], data_collator=data_collator
    )


def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not data_args.data_path:
        raise ValueError("--data_path is required and must name a saved train/dev DatasetDict")
    output = Path(training_args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not training_args.overwrite_output_dir:
            raise ValueError(
                "Output is not empty. Choose a new --output_dir or pass --overwrite_output_dir True"
            )

    torch_dtype = torch.bfloat16

    device_map = None
    quantization_config = None

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100:.4f}%")

    now = datetime.now()
    training_args.run_name = (training_args.run_name or "casesumm") + now.strftime(
        "_%Y-%m-%d_%H-%M"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset_text_field="prompt",
        max_seq_length=training_args.model_max_length,
        args=training_args,
        **data_module,
    )
    trainer.train()
    print(trainer.model)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

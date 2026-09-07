"""Load datasets for fine-tuning."""

from pathlib import Path
from typing import TYPE_CHECKING

from datasets import DatasetDict, load_from_disk

if TYPE_CHECKING:
    from datasets import Dataset


def load_hf_dataset(
    path: str | Path, structured: bool = True, split: str | None = None
) -> "DatasetDict | Dataset":
    """Load a saved dataset for fine-tuning.

    path: Directory created with save_to_disk, containing train and dev splits.
    structured: Keep cases whose syllabus contains "Held:" and whose opinion
        is more than ten tokens longer than the syllabus.
    split: Return just this split, or all splits when None.
    """
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Fine-tuning requires a saved DatasetDict with train and dev splits")
    if not {"train", "dev"} <= set(dataset):
        raise ValueError("Fine-tuning requires train and dev splits")
    if structured:
        dataset = dataset.filter(
            lambda row: (
                row["structured_syllabus"] == 1 and row["opinion_len"] - row["syllabus_len"] > 10
            )
        )
    if split is not None:
        return dataset[split]
    return dataset

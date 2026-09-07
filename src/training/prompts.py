"""Mistral prompt templates for training and inference."""

INSTR_END = "\n[/INST]\n"

PROMPT_DICT = {
    "prompt": (
        "<s>[INST]Provide a summary of the following Supreme Court opinion:\n\n"
        "[TEXT_START]\n\n{opinion}\n\n[TEXT_END]\n\n[/INST]\n{syllabus}</s>"
    ),
    "prompt_infer": (
        "<s>[INST]Provide a summary of the following Supreme Court opinion:\n\n"
        " [TEXT_START]\n\n{opinion}\n\n[TEXT_END]\n\n[/INST]"
    ),
}

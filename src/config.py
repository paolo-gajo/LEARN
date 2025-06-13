default_cfg = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt_layout_path": "./misc/prompt_layout.txt",
    "prompt_tags_path": "./misc/prompt_tags.txt",
    "verbose_output_path": "./pred_outputs",
    "suffix": ""
}

debug_cfg = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt_layout_path": "./misc/prompt_layout.txt",
    "prompt_tags_path": "./misc/prompt_tags.txt",
    "verbose_output_path": "./pred_outputs",
    "suffix": "",
    "train_steps": 5,
    "eval_steps": 5,
    "epochs": 1
}
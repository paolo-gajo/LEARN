import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data_utils import convert_files, CausalLMDataset
from eval_utils import Evaluator
from os_utils import get_time
from train_utils import set_seeds
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import argparse
import numpy as np
import copy
import os
import json

def main(args):
    set_seeds(args.seed)
    print(args)
    df = convert_files(args.data_path)
    print(df)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt_layout = open('./misc/prompt_layout.txt', 'r').read()
    prompt_tags = open('./misc/prompt_tags.txt', 'r').read()
    dataset = CausalLMDataset(df,
                            prompt_layout,
                            prompt_tags,
                            tokenizer,
                            n_icl_samples=args.n_icl_samples,
                            use_prompt_tags=args.use_prompt_tags,
                            seed=args.seed,
                            )
    dataset_train = Dataset.from_pandas(dataset.train_samples)

    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    if args.target_modules != 'full_ft':
        target_modules = [el+'_proj' for el in args.target_modules.split('-')]
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=False,
        )
        model = get_peft_model(model, lora_config)
    else:
        target_modules = args.target_modules
        lora_config = None
    
    # Create custom evaluator that has access to the expected outputs
    evaluator_dev = Evaluator(tokenizer, model)

    if not args.train_steps:
        args.train_steps = len(dataset.train_samples) // int(args.batch_size_train)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset_train,
        # eval_dataset=dataset_dev,
        # compute_metrics=evaluator.compute_metrics,
        peft_config=lora_config,
        args=SFTConfig(
            dataset_num_proc=1,
            packing=False,
            per_device_train_batch_size=args.batch_size_train,
            per_device_eval_batch_size=args.batch_size_eval,
            # eval_accumulation_steps=1,
            warmup_steps=5,
            max_steps=args.train_steps,
            # num_train_epochs=1,
            learning_rate=args.lr,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear", 
            seed=42,
            # output_dir="outputs",
            report_to="none",
            eval_strategy='no',
            # eval_steps=args.train_steps,
            # save_steps=args.train_steps,
            metric_for_best_model="f1",
            # load_best_model_at_end=True,
            label_names=["labels"],
        ),
    )
    results_dir = os.path.join("./results", f"{model_name.split('/')[-1]}_{get_time()}")
    os.makedirs(results_dir, exist_ok=True)
    max_val = -np.inf
    results_dev_list = []
    if args.eval_steps:
        dataset.dev_samples = dataset.dev_samples[:args.eval_steps]
    for epoch in range(args.epochs):
        model.train()
        trainer.train()
        results_dev = evaluator_dev.evaluate(dataset.dev_samples,
                                     epoch,
                                     args.batch_size_eval,
                                     verbose = args.verbose_eval,
                                     split='dev',
                                     )
        print(f'dev @ {epoch + 1} epochs:', results_dev)
        results_dev_list.append(results_dev)
        json_path_dev = os.path.join(results_dir, 'results_dev.json')
        
        with open(json_path_dev, 'w', encoding='utf8') as f:
            json.dump(results_dev_list, f, ensure_ascii = False, indent = 4)
        if results_dev['micro_f1'] > max_val:
            max_val = results_dev['micro_f1']
            best_model = copy.deepcopy(model)
    if args.eval_steps:
        dataset.test_samples = dataset.test_samples[:args.eval_steps]
    evaluator_test = Evaluator(tokenizer, best_model)
    results_test = evaluator_test.evaluate(dataset.test_samples,
                                     epoch,
                                     args.batch_size_eval,
                                     verbose = args.verbose_eval,
                                     split='test',
                                     )
    print(f'test @ {epoch + 1} epochs:', results_test)

    json_path_test = os.path.join(results_dir, 'results_test.json')
    with open(json_path_test, 'w', encoding='utf8') as f:
        json.dump(results_test, f, ensure_ascii = False, indent = 4)

    model_dir = f"./models/{model_name.split('/')[-1]}_{get_time()}"
    best_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal language modeling trainer")
    parser.add_argument("--data_path", help="Directory path containing input files", default='./data/')
    parser.add_argument("--lr", type=float, help="Learning rate", default=2e-4)
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=0)
    parser.add_argument("--eval_steps", type=int, help="Number of training steps", default=0)
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=3)
    parser.add_argument("--n_icl_samples", type=int, help="Number of training epochs", default=10)
    parser.add_argument("--use_prompt_tags", type=int, help="Whether to include tags in the prompt (0 or 1 as int acting as bool)", default=1)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=4)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=4)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--target_modules", type=str, help="List of LoRA modules to use (as dash-separated string).", default='q-k-v')
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--verbose_eval", action="store_true", help="Enable verbose evaluation output")
    parser.add_argument("--layout_path", help="Path of the layout text file", default='./misc/prompt_layout_tags.txt')
    parser.add_argument("--tags_path", help="Path of the tags text file", default='./misc/prompt_tags.txt')
    parser.add_argument("--seed", type=int, help="Gradient accumulation steps", default=42)
    args = parser.parse_args()
    main(args)
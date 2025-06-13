import torch
import argparse
import numpy as np
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data_utils import convert_files, CausalLMDataset
from eval_utils import Evaluator
from os_utils import setup_config
from train_utils import set_seeds
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

def main(args):
    config = setup_config(args, json.load(open(args.config_path, 'r')))
    print(config)
    results_dir = os.path.join("./results", f"{config['model_name'].split('/')[-1]}", config['suffix'])
    print(f'Will save results to: {results_dir}')
    os.makedirs(results_dir, exist_ok=True)

    set_seeds(config['seed'])
    df = convert_files(config['data_path'])
    df.to_csv('./misc/debug_df.csv')
    print(df)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'],
                                              padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    

    dataset = CausalLMDataset(df,
                            tokenizer,
                            config,
                            )
    dataset.train_samples.to_csv('./misc/debug_train.csv')
    dataset_train = Dataset.from_pandas(dataset.train_samples)

    if config['load_in_4bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif config['load_in_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    if args.do_train and config['target_modules'] != 'full_ft':
        if not config['train_steps']:
            config['train_steps'] = len(dataset.train_samples) // int(config['batch_size_train'])
        target_modules = [el+'_proj' for el in config['target_modules'].split('-')]
        if config['load_in_4bit'] or config['load_in_8bit']:
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
        print('Applied LoRA!')
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
                per_device_train_batch_size=config['batch_size_train'],
                per_device_eval_batch_size=config['batch_size_eval'],
                # eval_accumulation_steps=1,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs = {"use_reentrant": False},
                warmup_steps=5,
                max_steps=config['train_steps'],
                # num_train_epochs=1,
                learning_rate=config['lr'],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear", 
                seed=config['seed'],
                # output_dir="outputs",
                report_to="none",
                eval_strategy='no',
                # eval_steps=config['eval_steps'],
                # save_steps=config['train_steps'] // 5,
                metric_for_best_model="f1",
                # load_best_model_at_end=True,
                label_names=["labels"],
            ),
        )
    
    # Create custom evaluator that has access to the expected outputs
    evaluator_dev = Evaluator(tokenizer, model, config)

    
    max_val = -np.inf
    best_epoch = 0
    results_dev_list = []
    
    if args.do_train:
        model_dir = os.path.join("./models/", config['model_name'].split('/')[-1], config['suffix'])
        print(f'Training, will save models to: {model_dir}')
        print('VRAM usage:',torch.cuda.memory_allocated())
        for epoch in range(config['epochs']):
            model.train()
            trainer.train()
            results_dev = evaluator_dev.evaluate(dataset.dev_samples,
                                        epoch,
                                        config['batch_size_eval'],
                                        verbose = config['verbose_eval'],
                                        split='dev',
                                        )
            print(f'dev @ {epoch + 1} epochs:', results_dev)
            results_dev_list.append(results_dev)
            json_path_dev = os.path.join(results_dir, 'results_dev.json')
            
            with open(json_path_dev, 'w', encoding='utf8') as f:
                json.dump(results_dev_list, f, ensure_ascii = False, indent = 4)
            if results_dev['micro_f1'] > max_val:
                max_val = results_dev['micro_f1']
                model.save_pretrained(model_dir, save_adapter=True, save_config=True)
                print(f"Best model updated with micro F1 = {results_dev['micro_f1']}")
                print(f"Best model saved to: {model_dir}")
                tokenizer.save_pretrained(model_dir)
                best_epoch = epoch
        
        print('VRAM usage:',torch.cuda.memory_allocated())
        print('Model deleted, test time...')
        model.to('cpu')
        del model
        torch.cuda.empty_cache()
        print('VRAM usage:',torch.cuda.memory_allocated())
        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                        quantization_config=quantization_config,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map='auto',
                                                        trust_remote_code=True)

        evaluator_test = Evaluator(tokenizer, model, config)
        results_test = evaluator_test.evaluate(dataset.test_samples,
                                     best_epoch,
                                     config['batch_size_eval'],
                                     verbose = config['verbose_eval'],
                                     split='test',
                                     )
    print(f'test @ {best_epoch + 1} epochs:', results_test)

    json_path_test = os.path.join(results_dir, 'results_test.json')
    with open(json_path_test, 'w', encoding='utf8') as f:
        json.dump(results_test, f, ensure_ascii = False, indent = 4)
    
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w', encoding='utf8') as f:
        json.dump(config, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal language modeling trainer")
    parser.add_argument("--data_path", type=str, help="Directory path containing input files", default='./data/')
    parser.add_argument("--model_name", type=str, help="HF name or local model path", default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--lr", type=float, help="Learning rate", default=2e-4)
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=0)
    parser.add_argument("--eval_steps", type=int, help="Number of training steps", default=0)
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=3)
    parser.add_argument("--n_icl_samples", type=int, help="Number of training epochs", default=10)
    parser.add_argument("--use_prompt_tags", type=int, help="Whether to include tags in the prompt (0 or 1 as int acting as bool)", default=1)
    parser.add_argument("--do_train", type=int, help="Whether to train the model", default=1)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=4)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=4)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--target_modules", type=str, help="List of LoRA modules to use (as dash-separated string).", default='q-k-v')
    parser.add_argument("--load_in_4bit", type=int, help="Use 4-bit quantization", default=0)
    parser.add_argument("--load_in_8bit", type=int, help="Use 8-bit quantization", default=0)
    parser.add_argument("--verbose_eval", type=int, help="Enable verbose evaluation output", default=0)
    parser.add_argument("--prompt_layout_path", type=str, help="Path of the layout text file", default='./misc/prompt_layout.txt')
    parser.add_argument("--config_path", type=str, help="Path of the config file", default='./misc/default_cfg.json')
    parser.add_argument("--tag_dict_path", type=str, help="Path of the coarse tag dictionary", default='./misc/coarse_tags.json')
    parser.add_argument("--coarse", type=int, help="Whether to use coarse tags", default=0)
    parser.add_argument("--seed", type=int, help="Seed used for the experiments", default=42)
    parser.add_argument("--suffix", type=str, help="Path of the tags text file", default='')
    
    args = parser.parse_args()
    main(args)
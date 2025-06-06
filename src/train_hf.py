import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from convert import convert_files
from utils import CompletionDataset
from datasets import Dataset

from trl import SFTConfig, SFTTrainer
import argparse

def main():
    df = convert_files(args.data_path)
    print(df)

    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left')
    tokenizer.pad_token = tokenizer.eos_token
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prompt_layout = open('./misc/prompt_layout_no_tags.txt', 'r').read()
    prompt_tags = open('./misc/prompt_tags.txt', 'r').read()
    dataset = CompletionDataset(df, prompt_layout, prompt_tags, tokenizer)
    dataset_train = Dataset.from_list(dataset.train)
    dataset_dev = Dataset.from_list(dataset.dev)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    def compute_metrics(sample):
        return ...

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        compute_metrics=compute_metrics,
        # peft_config=lora_config,
        args=SFTConfig(
            # dataset_text_field="text",
            completion_only_loss=True,
            max_seq_length=args.max_seq_length,
            dataset_num_proc=2,
            packing=True,
            per_device_train_batch_size=args.batch_size_train,
            per_device_eval_batch_size=args.batch_size_eval,
            # gradient_accumulation_steps=args.grad_acc_steps,
            # gradient_checkpointing_kwargs={'use_reentrant':False},
            # gradient_checkpointing=True,
            warmup_steps=5,
            max_steps=args.train_steps,
            learning_rate=args.lr,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
            report_to="none",
            eval_strategy='steps',
            eval_steps=args.train_steps,
            save_steps=args.train_steps,
            metric_for_best_model="f1",
            load_best_model_at_end=True,
            use_liger_kernel=True,
        ),
    )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal language modeling trainer")
    parser.add_argument("--data_path", help="Directory path containing input files", default='./data/')
    parser.add_argument("--lr", type=float, help="Learning rate", default=2e-4)
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=200)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=1)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=1)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    args = parser.parse_args()
    main()
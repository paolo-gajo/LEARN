import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from convert import convert_files
from utils import CompletionDataset, get_time
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import argparse
from sklearn.metrics import f1_score

def main(args):
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
    dataset_dev = Dataset.from_list(dataset.dev[:10])

    # Configure quantization if needed
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
        device_map='auto',
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

    def compute_metrics(sample):
        label_ids = sample.label_ids
        label_mask = label_ids != -100
        labels = label_ids[label_mask]
        print('#' * 100)
        print(tokenizer.batch_decode(labels))
        preds = sample.predictions.argmax(axis=-1)
        preds_valid = preds[label_mask]
        print('#' * 100)
        for i, el in enumerate(preds):
            print(i, tokenizer.batch_decode(el))
            print()
        print('#' * 100)
        print(tokenizer.batch_decode(preds_valid))
        metrics_dict = {'f1': f1_score(labels, preds_valid, average='micro')}
        return metrics_dict
    
    def concat_prompt_completion(example):
        return {"text": example["prompt"] + example["completion"]}

    dataset_train = dataset_train.map(concat_prompt_completion, remove_columns=["prompt", "completion"])
    dataset_dev = dataset_dev.map(concat_prompt_completion, remove_columns=["prompt", "completion"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        compute_metrics=compute_metrics,
        # peft_config=lora_config,
        args=SFTConfig(
            dataset_text_field="text",
            # completion_only_loss=True,
            max_seq_length=args.max_seq_length,
            dataset_num_proc=1,
            packing=False,
            per_device_train_batch_size=args.batch_size_train,
            per_device_eval_batch_size=args.batch_size_eval,
            # gradient_accumulation_steps=args.grad_acc_steps,
            # gradient_checkpointing_kwargs={'use_reentrant':False},
            # gradient_checkpointing=True,
            eval_accumulation_steps=1,
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
            # use_liger_kernel=True,
            label_names=["labels"],
        ),
    )
    trainer.train()
    save_dir = f"./models/{model_name.split('/')[-1]}_{get_time()}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal language modeling trainer")
    parser.add_argument("--data_path", help="Directory path containing input files", default='./data/')
    parser.add_argument("--lr", type=float, help="Learning rate", default=2e-4)
    parser.add_argument("--train_steps", type=int, help="Number of training steps", default=100)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=4)
    parser.add_argument("--batch_size_eval", type=int, help="Batch size for evaluation", default=1)
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--target_modules", type=str, help="List of LoRA modules to use (as dash-separated string).", default='q-k-v')
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    args = parser.parse_args()
    main(args)
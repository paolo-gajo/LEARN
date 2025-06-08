from datasets import Dataset
import pandas as pd
import re
from typing import List
from random import shuffle
from datetime import datetime
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm
import os
from collections import defaultdict

class CausalLMDataset:
    def __init__(self,
                data: pd.DataFrame,
                prompt_layout: str,
                prompt_tags: str,
                tokenizer,
                n_icl_samples: int=3, 
                clean: bool=True,
                ):
        self.data = data
        self.prompt_layout = prompt_layout
        self.prompt_tags = prompt_tags
        self.tokenizer = tokenizer
        self.n_icl_samples = n_icl_samples
        if clean:
            self.clean()
        self.sys_prompt = 'You are an AI specialized in the task of annotating grammatical errors.'
        df_train, df_dev = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train = df_train.reset_index(drop=True)
        df_dev, df_test = train_test_split(df_dev, test_size=0.5, random_state=42)
        self.dev = df_dev.reset_index(drop=True)
        self.test = df_test.reset_index(drop=True)
        self.train_samples = self.make_samples('train')
        self.dev_samples = self.make_samples('dev')
        self.test_samples = self.make_samples('test')

    def clean(self):
        self.data['text_og'] = self.data['text_og'].apply(lambda x: x.replace(r'\0', ''))
    
    def make_samples(self, split):
        split_data = getattr(self, split)
        text_og_list = split_data['text_og']
        text_an_list = split_data['text_an']
        text_og_list_examples = self.train['text_og']
        text_an_list_examples = self.train['text_an']
        chat_list = []
        for i in range(len(text_og_list)):
            sentence = text_og_list.iloc[i]
            expected_output = text_an_list.iloc[i]
            
            if split == 'train':
                examples_an = text_an_list_examples.drop(labels=split_data.index[i])
            else:
                examples_an = text_an_list_examples
            mask_pos = examples_an.apply(lambda x: '</' in x)
            mask_neg = examples_an.apply(lambda x: '</' not in x)
            examples_an_pos = examples_an[mask_pos].sample(n=self.n_icl_samples)
            examples_an_neg = examples_an[mask_neg].sample(n=self.n_icl_samples)
            examples_og_pos = text_og_list_examples[examples_an_pos.index]
            examples_og_neg = text_og_list_examples[examples_an_neg.index]
            examples_pos = [f'{og}###{an}' for og, an in zip(examples_og_pos, examples_an_pos)]
            examples_neg = [f'{og}###{an}' for og, an in zip(examples_og_neg, examples_an_neg)]
            examples = examples_pos + examples_neg
            shuffle(examples)
            examples = '\n'.join(examples)
            prompt = self.prompt_layout.format(prompt_tags=self.prompt_tags,
                                               examples=examples,
                                               sentence=sentence,
                                               eos_token=self.tokenizer.eos_token)
            
            # Create the prompt part (same for train and eval)
            chat_prompt = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ]
            
            if split == 'train':
                # For training: include the assistant response
                chat = chat_prompt + [{"role": "assistant", "content": expected_output}]
                chat_formatted = self.tokenizer.apply_chat_template(chat,
                                                        tokenize=False,
                                                        add_generation_prompt=False)
                chat_list.append({'text': chat_formatted})
            else:
                # For evaluation: separate prompt and expected output
                chat_prompt_formatted = self.tokenizer.apply_chat_template(chat_prompt,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
                chat_list.append({
                    'prompt': chat_prompt_formatted,
                    'completion': expected_output,
                    'prompt_length': len(self.tokenizer.encode(chat_prompt_formatted))
                })
        return pd.DataFrame(chat_list)  
    
def make_tags_prompt(tags_csv_path: str):
    df = pd.read_csv(tags_csv_path, sep='\t')
    tags = df['Tag'].tolist()
    description = df['Description'].tolist()
    out_prompt = ''
    for t, d in zip(tags, description):
        out_prompt += f'{t}: {d}\n'
    return out_prompt

def get_time():
    return str(datetime.now()).split('.')[0].replace(' ', '').replace('-', '').replace(':', '')[2:]

class Evaluator:
    def __init__(self,
                 tokenizer,
                 model,
                 verbose_output_path='./eval_outputs',
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = self.model.base_model.model.config.name_or_path.replace('/', '-')
        self.verbose_output_path = os.path.join(verbose_output_path,
                                                self.model_name,
                                                get_time(),)
        os.makedirs(self.verbose_output_path, exist_ok=True)

    def evaluate(self,
                 df,
                 epoch,
                 batch_size=1,
                 verbose=False,
                 split='dev',
                 ):
        """
        Evaluate the model on the test set using a configurable `batch_size`.
        This uses a unified run_model(...) method that can handle both
        single-item or multi-item batches.
        """
        trues = []
        preds = []

        prompt_list = df['prompt'].to_list()
        comp_list = df['completion'].to_list()

        # # Gather all possible relation types for eventual per-relation metrics
        # all_true_rel_types = set([el[1] for t_list in comp_list for el in t_list])
        # rel_type_preds_trues = {k: {'trues': [], 'preds': []} for k in all_true_rel_types}

        # Progress bar over sub-batches, rather than item by item
        pbar = tqdm(range(0, len(df), batch_size), desc=f'Model: {self.model_name}')
        self.t_counter = 0  # initialize model timeout counter

        verbose_output = ''

        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, len(df))
            batch_texts = prompt_list[start_idx:end_idx]
            batch_completions = comp_list[start_idx:end_idx]

            # Run the model on the batch. If batch_size=1, this will still work.
            outputs = self.run_model(batch_texts)

            # If run_model returned a single string (when batch_size=1), make it a list
            if isinstance(outputs, str):
                outputs = [outputs]

            # Now iterate over each sample in the batch
            for text, completion, output in zip(batch_texts, batch_completions, outputs):
                true_list = self.extract_tags(completion)
                pred_list = self.extract_tags(output)
                print('completion', completion, '-->', true_list)
                print('output', output, '-->', pred_list)
                trues.append(true_list)
                preds.append(pred_list)

                # # Collect per-relation predictions + gold
                # for rel_type in all_true_rel_types:
                #     trues_type = [t for t in true_list if t[1] == rel_type]
                #     preds_type = [p for p in pred_list if p[1] == rel_type]
                #     rel_type_preds_trues[rel_type]['trues'].append(trues_type)
                #     rel_type_preds_trues[rel_type]['preds'].append(preds_type)

                # Optionally display partial F1 and log verbose output
                if verbose:
                    metrics_sample = calculate_metrics([true_list], [pred_list])
                    metrics_current = calculate_metrics(trues, preds)
                    pbar.set_description(f"Last F1: {round(metrics_sample['macro_f1'], 2)}\nOverall F1: {round(metrics_current['macro_f1'], 2)}")

                    output_texts = (
                        '\n' + f'text: {text}' +
                        '\n' + f'result: {output}' +
                        '\n' + f'pred: {pred_list}' +
                        '\n' + f'trues: {true_list}'
                    )
                    output_metrics = (
                        '\n' + f'metrics_sample: {metrics_sample}' +
                        '\n' + f'metrics_current: {metrics_current}'
                    )

                    verbose_output += output_texts + output_metrics

        # If verbose output path is specified, save the accumulated logs
        if verbose:
            verbose_output = verbose_output.lstrip()
            save_path = os.path.join(self.verbose_output_path, f'results_{split}_{epoch + 1}.log')
            with open(save_path, 'w', encoding='utf8') as f:
                f.write(verbose_output)
        # all_pred_rel_types = set([p[1] for p_list in preds for p in p_list])
        # print(f'All types of predicted relations: {all_pred_rel_types}')
        # Final strict micro-averaged metrics

        # # Per-relation metrics
        # rel_type_metrics = {k: {} for k in all_true_rel_types}
        # for rel_type, pr_dict in rel_type_preds_trues.items():
        #     p_rel, r_rel, f1_rel = calculate_metrics(pr_dict['trues'], pr_dict['preds'])
        #     rel_type_metrics[rel_type]['precision'] = p_rel
        #     rel_type_metrics[rel_type]['recall']    = r_rel
        #     rel_type_metrics[rel_type]['f1']        = f1_rel

        return calculate_metrics(trues, preds)

    def run_model(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_token_type_ids=False,
        ).to(self.model.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=100
            )

        # Decode each output
        decoded = []
        input_lengths = [len(seq) for seq in tokenized["input_ids"]]

        # Then slice correctly:
        for i, seq in enumerate(outputs):
            if isinstance(input_lengths, list):
                in_len = input_lengths[i]
            else:
                in_len = input_lengths
            gen_tokens = seq[in_len:]
            decoded_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            decoded.append(decoded_text)

        # If single text was provided, return just the string; else return a list
        return decoded[0] if len(decoded) == 1 else decoded

    @staticmethod
    def extract_tags(input: str) -> List[List[str]]:
        return re.findall(r'<(\w+)[^>]*>(.*?)</\1>', input)

def calculate_metrics(sample_trues, sample_preds):
    """
    Calculate metrics where:
    - Exact match requires both tag and source text to match
    - Final metrics are aggregated by tag type only
    """
    if len(sample_trues) != len(sample_preds):
        raise ValueError("Length of true_tags and pred_tags must match")
    
    # Initialize counters by tag type
    dict_tp = defaultdict(int)
    dict_fp = defaultdict(int)
    dict_fn = defaultdict(int)
    
    # Count TP, FP, FN
    for true_tags, pred_tags in zip(sample_trues, sample_preds):
        # Pad shorter list with ('O', '') tuples
        max_len = max(len(true_tags), len(pred_tags))
        true_tags_padded = true_tags + [('O', '')] * (max_len - len(true_tags))
        pred_tags_padded = pred_tags + [('O', '')] * (max_len - len(pred_tags))
        
        for true_tuple, pred_tuple in zip(true_tags_padded, pred_tags_padded):
            true_tag = true_tuple[0] if isinstance(true_tuple, tuple) else true_tuple
            pred_tag = pred_tuple[0] if isinstance(pred_tuple, tuple) else pred_tuple
            
            # Exact match: both tag and source must match
            if true_tuple == pred_tuple:
                dict_tp[true_tag] += 1  # Count TP for the tag type
            else:
                dict_fp[pred_tag] += 1  # Count FP for predicted tag type
                dict_fn[true_tag] += 1  # Count FN for true tag type
    
    # Calculate micro-averaged metrics
    total_tp = sum(dict_tp.values())
    total_fp = sum(dict_fp.values())
    total_fn = sum(dict_fn.values())
    
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall > 0 else 0)
    
    # Calculate macro-averaged metrics
    all_tags = set()
    for true_tags, pred_tags in zip(sample_trues, sample_preds):
        for true_tuple in true_tags:
            all_tags.add(true_tuple[0] if isinstance(true_tuple, tuple) else true_tuple)
        for pred_tuple in pred_tags:
            all_tags.add(pred_tuple[0] if isinstance(pred_tuple, tuple) else pred_tuple)
    
    precisions, recalls, f1s = [], [], []
    for tag in all_tags:
        precision = dict_tp[tag] / (dict_tp[tag] + dict_fp[tag]) if dict_tp[tag] + dict_fp[tag] > 0 else 0
        recall = dict_tp[tag] / (dict_tp[tag] + dict_fn[tag]) if dict_tp[tag] + dict_fn[tag] > 0 else 0
        f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    macro_precision = sum(precisions) / len(precisions) if precisions else 0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0
    
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }
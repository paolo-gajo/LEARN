from collections import Counter, defaultdict, OrderedDict
import re
import os
from os_utils import get_time
from tqdm.auto import tqdm
import torch
from typing import List

class Evaluator:
    def __init__(self,
                 tokenizer = None,
                 model = None,
                 config: dict = None,
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        if self.model is not None:
            self.verbose_output_path = os.path.join(config['verbose_output_path'],
                                                    config['model_name'].replace('/', '-'),
                                                    get_time(),)

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

        # Progress bar over sub-batches, rather than item by item
        pbar = tqdm(range(0, len(df), batch_size), desc=f"Model: {self.config['model_name']}")
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
                trues.append(true_list)
                preds.append(pred_list)

                # Optionally display partial F1 and log verbose output
                if verbose:
                    print('completion', completion, '-->', true_list)
                    print('output', output, '-->', pred_list)
                    metrics_sample = self.calculate_metrics([true_list], [pred_list])
                    metrics_current = self.calculate_metrics(trues, preds)
                    pbar.set_description(f"Overall/Last F1: ({round(metrics_current['macro_f1'], 2)}, {round(metrics_sample['macro_f1'], 2)})")

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
            os.makedirs(self.verbose_output_path, exist_ok=True)
            save_path = os.path.join(self.verbose_output_path, f'results_{split}_{epoch + 1}.log')
            with open(save_path, 'w', encoding='utf8') as f:
                f.write(verbose_output)
        out_dict = self.calculate_metrics(trues, preds)
        out_dict['epoch'] = epoch
        return out_dict

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
        """
        Extract tags with 'corr' attribute.
        Returns list of tuples: (tag, content)
        """
        # Primary pattern: corr with double quotes (with proper closing tag)
        pattern1 = r'<([A-Z]+[A-Z0-9]*)\s+[^>]*corr="[^"]*"[^>]*>(.*?)</\1>'
        matches1 = re.findall(pattern1, input)
        
        # Secondary pattern: corr with single quotes (with proper closing tag)  
        pattern2 = r"<([A-Z]+[A-Z0-9]*)\s+[^>]*corr='[^']*'[^>]*>(.*?)</\1>"
        matches2 = re.findall(pattern2, input)
        
        # Combine all matches (preserving duplicates)
        all_matches = matches1 + matches2
        
        return [(tag, content.strip()) for tag, content in all_matches]

    @staticmethod
    def calculate_metrics(sample_trues, sample_preds):
        """
        Calculate metrics preserving duplicate (tag, content) pairs.
        Metrics are calculated per tag type, but duplicate spans are preserved.
        """
        
        dict_tp = defaultdict(int)
        dict_fp = defaultdict(int)
        dict_fn = defaultdict(int)
        
        for true_tags, pred_tags in zip(sample_trues, sample_preds):
            # Count occurrences of each (tag, content) pair
            counts_true = Counter(true_tags)
            counts_pred = Counter(pred_tags)
            
            # Get all unique (tag, content) combinations from both true and pred
            all_pairs = set(true_tags) | set(pred_tags)
            
            for tag_content_pair in all_pairs:
                tag_name = tag_content_pair[0]
                count_true = counts_true[tag_content_pair]
                count_pred = counts_pred[tag_content_pair]
                
                # True Positives: minimum of true and predicted counts
                tp = min(count_true, count_pred)
                dict_tp[tag_name] += tp
                
                # False Positives: predicted more than true
                fp = max(0, count_pred - count_true)
                dict_fp[tag_name] += fp
                
                # False Negatives: true more than predicted  
                fn = max(0, count_true - count_pred)
                dict_fn[tag_name] += fn
        
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
            for tag, content in true_tags:
                all_tags.add(tag)
            for tag, content in pred_tags:
                all_tags.add(tag)
        
        precisions, recalls, f1s = [], [], []
        tag_dict = defaultdict(dict)
        for tag in all_tags:
            precision = dict_tp[tag] / (dict_tp[tag] + dict_fp[tag]) if dict_tp[tag] + dict_fp[tag] > 0 else 0
            recall = dict_tp[tag] / (dict_tp[tag] + dict_fn[tag]) if dict_tp[tag] + dict_fn[tag] > 0 else 0
            f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            tag_dict[tag]['precision'] = precision
            tag_dict[tag]['recall'] = recall
            tag_dict[tag]['f1'] = f1
        tag_dict = OrderedDict(sorted(tag_dict.items(), key=lambda x: x[1]['f1'], reverse=True))
        macro_precision = sum(precisions) / len(precisions) if precisions else 0
        macro_recall = sum(recalls) / len(recalls) if recalls else 0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0
        
        return {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_tag_metrics": tag_dict,
        }

if __name__ == "__main__":
    evaluator = Evaluator()
    gold = [
        # [('WO', 'do to'), ('FS', 'excercice'), ('GPR', 'that'), ('XVCO', 'would reserve'), ('GA', '\\0'), ('GNN', 'question')],
        [('WO', 'do to'), ('FS', 'excercice'), ('GPR', 'that'), ('XVCO', 'would reserve'), ('GA', '\\0'), ('GNN', 'question')],
        ]
    pred = [
        # [('GVAUX', 'do'), ('WO', 'to do'), ('FS', 'excercice'), ('GADJO', 'encouraging'), ('GPP', 'tutor'), ('GVT', 'helps')],
        [('DMCC', 'Hello!'), ('GVM', 'do'), ('DMCC', 'to'), ('FS', 'excercice')],
        ]
    for g, p in zip(gold, pred):
        metrics = evaluator.calculate_metrics([g], [p])
        print(metrics)
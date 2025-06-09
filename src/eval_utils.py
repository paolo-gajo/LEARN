from collections import Counter, defaultdict

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

if __name__ == "__main__":
    gold = [
        # [('WO', 'do to'), ('FS', 'excercice'), ('GPR', 'that'), ('XVCO', 'would reserve'), ('GA', '\\0'), ('GNN', 'question')],
        [('WO', 'do to'), ('FS', 'excercice'), ('GPR', 'that'), ('XVCO', 'would reserve'), ('GA', '\\0'), ('GNN', 'question')],
        ]
    pred = [
        # [('GVAUX', 'do'), ('WO', 'to do'), ('FS', 'excercice'), ('GADJO', 'encouraging'), ('GPP', 'tutor'), ('GVT', 'helps')],
        [('DMCC', 'Hello!'), ('GVM', 'do'), ('DMCC', 'to'), ('FS', 'excercice')],
        ]
    for g, p in zip(gold, pred):
        metrics = calculate_metrics([g], [p])
        print(metrics)
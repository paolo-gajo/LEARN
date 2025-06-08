from collections import defaultdict

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

if __name__ == "__main__":
    gold = [('WO', 'do to'), ('FS', 'excercice'), ('GPR', 'that'), ('XVCO', 'would reserve'), ('GA', '\\0'), ('GNN', 'question')]
    pred = [('GVAUX', 'do'), ('WO', 'to do'), ('FS', 'excercice'), ('GADJO', 'encouraging'), ('GPP', 'tutor'), ('GVT', 'helps')]

    metrics = calculate_metrics([gold], [pred])
    print(metrics)
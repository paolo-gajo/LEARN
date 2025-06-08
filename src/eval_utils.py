from collections import defaultdict

def calculate_metrics(sample_trues, sample_preds):
    """
    Calculate metrics using content-based matching, ignoring positions.
    Position is only used to prevent set() from removing duplicates.
    """
    dict_tp = defaultdict(int)
    dict_fp = defaultdict(int)
    dict_fn = defaultdict(int)
    
    for true_tags, pred_tags in zip(sample_trues, sample_preds):
        # Convert to (tag, content) pairs, ignoring position for comparison
        true_pairs = set((tag, content) for tag, content, pos in true_tags)
        pred_pairs = set((tag, content) for tag, content, pos in pred_tags)
        
        # True Positives
        tp_pairs = true_pairs & pred_pairs
        for tag_name, content in tp_pairs:
            dict_tp[tag_name] += 1
        
        # False Positives
        fp_pairs = pred_pairs - true_pairs
        for tag_name, content in fp_pairs:
            dict_fp[tag_name] += 1
        
        # False Negatives
        fn_pairs = true_pairs - pred_pairs
        for tag_name, content in fn_pairs:
            dict_fn[tag_name] += 1
    
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
        for tag, content, pos in true_tags:
            all_tags.add(tag)
        for tag, content, pos in pred_tags:
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
    gold = [('WO', 'do to', 0), ('FS', 'excercice', 1), ('GPR', 'that', 2), ('XVCO', 'would reserve', 3), ('GA', '\\0', 4), ('GNN', 'question', 5)]
    pred = [('GVAUX', 'do', 0), ('WO', 'to do', 1), ('FS', 'excercice', 2), ('GADJO', 'encouraging', 3), ('GPP', 'tutor', 4), ('GVT', 'helps', 5)]

    metrics = calculate_metrics([gold], [pred])
    print(metrics)
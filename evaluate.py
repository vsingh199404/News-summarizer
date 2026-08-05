import argparse
from typing import List, Dict, Any

def compute_rouge(prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE scores using rouge_scorer."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Error: rouge_score is not installed. Install with 'pip install rouge_score'")
        return {}
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    result = {}
    for metric, score in scores.items():
        result[metric] = {
            'precision': round(score.precision, 4),
            'recall': round(score.recall, 4),
            'f1': round(score.fmeasure, 4)
        }
    return result

def compute_rouge_batch(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """Average ROUGE scores over lists."""
    if not predictions or not references or len(predictions) != len(references):
        return {}
        
    avg_scores = {
        'rouge1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        'rouge2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        'rougeL': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    }
    
    valid_count = 0
    for pred, ref in zip(predictions, references):
        if not ref.strip():
            continue
        scores = compute_rouge(pred, ref)
        if not scores:
            continue
            
        valid_count += 1
        for metric in avg_scores:
            for stat in avg_scores[metric]:
                avg_scores[metric][stat] += scores[metric][stat]
                
    if valid_count > 0:
        for metric in avg_scores:
            for stat in avg_scores[metric]:
                avg_scores[metric][stat] = round(avg_scores[metric][stat] / valid_count, 4)
                
    return avg_scores

def format_rouge_table(scores: Dict[str, Dict[str, float]]) -> str:
    """Format scores as a table."""
    if not scores:
        return "No scores available."
        
    header = f"{'Metric':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}"
    separator = "-" * len(header)
    
    rows = [header, separator]
    for metric, stats in scores.items():
        row = f"{metric:<10} | {stats['precision']:<10.4f} | {stats['recall']:<10.4f} | {stats['f1']:<10.4f}"
        rows.append(row)
        
    return "\n".join(rows)

def evaluate_model(model: Any, articles: List[str], references: List[str], verbose: bool = True):
    """Full pipeline: generate summaries, compute ROUGE, print results."""
    from model import NewsSummarizer # local import to avoid circular dependency
    
    if not isinstance(model, NewsSummarizer):
        raise ValueError("Model must be an instance of NewsSummarizer")
        
    print(f"Evaluating model on {len(articles)} samples...")
    predictions = []
    
    for i, article in enumerate(articles):
        pred = model.summarize(article)
        predictions.append(pred)
        
        if verbose:
            print(f"\nSample {i+1}:")
            print(f"Reference: {references[i]}")
            print(f"Prediction: {pred}")
            scores = compute_rouge(pred, references[i])
            print(format_rouge_table(scores))
            
    print("\nOverall Batch Results:")
    batch_scores = compute_rouge_batch(predictions, references)
    print(format_rouge_table(batch_scores))
    return batch_scores

if __name__ == "__main__":
    print("Testing Evaluation Module...")
    
    preds = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast fox jumped over a sleepy dog."
    ]
    refs = [
        "A quick brown fox jumps over a lazy dog.",
        "The fast fox leaps over the sleeping dog."
    ]
    
    print("\nSingle pair evaluation (Pair 1):")
    scores1 = compute_rouge(preds[0], refs[0])
    print(format_rouge_table(scores1))
    
    print("\nBatch evaluation:")
    batch_scores = compute_rouge_batch(preds, refs)
    print(format_rouge_table(batch_scores))

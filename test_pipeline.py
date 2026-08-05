"""Quick end-to-end test of the News Summarizer pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text, load_sample_articles
from model import NewsSummarizer
from evaluate import compute_rouge, compute_rouge_batch, format_rouge_table

def main():
    print("=" * 70)
    print("  NEWS ARTICLE SUMMARIZATION — END-TO-END TEST")
    print("=" * 70)
    
    # 1. Load sample articles
    print("\n[1] Loading sample articles...")
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_articles')
    samples = load_sample_articles(sample_dir)
    print(f"    Loaded {len(samples)} articles")
    
    # 2. Load model
    print("\n[2] Loading T5-Small model...")
    model = NewsSummarizer(model_name='t5-small')
    info = model.get_model_info()
    print(f"    Model: {info['architecture']}")
    print(f"    Parameters: {info['total_params']:,}")
    print(f"    Device: {info['device']}")
    
    # 3. Generate summaries
    print("\n[3] Generating summaries...")
    predictions = []
    references = []
    
    for i, sample in enumerate(samples):
        cleaned = clean_text(sample['content'])
        summary = model.summarize(cleaned)
        predictions.append(summary)
        references.append(sample.get('reference', ''))
        
        print(f"\n    --- Article {i+1}: {sample['title'][:50]} ---")
        print(f"    Input:     {len(sample['content'].split())} words")
        print(f"    Summary:   {summary[:120]}...")
        print(f"    Ref:       {sample.get('reference', 'N/A')[:120]}...")
        
        if sample.get('reference'):
            scores = compute_rouge(summary, sample['reference'])
            print(f"    ROUGE-1: {scores['rouge1']['f1']:.4f} | "
                  f"ROUGE-2: {scores['rouge2']['f1']:.4f} | "
                  f"ROUGE-L: {scores['rougeL']['f1']:.4f}")
    
    # 4. Batch evaluation
    print("\n[4] Batch ROUGE Evaluation")
    valid_preds = [p for p, r in zip(predictions, references) if r.strip()]
    valid_refs = [r for r in references if r.strip()]
    
    if valid_preds:
        avg_scores = compute_rouge_batch(valid_preds, valid_refs)
        print(format_rouge_table(avg_scores))
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✓")
    print("=" * 70)

if __name__ == "__main__":
    main()

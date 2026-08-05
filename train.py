import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import time

from preprocess import load_cnn_dailymail, preprocess_dataset
from model import NewsSummarizer

def train(args):
    print(f"Training parameters: {args}")
    
    # Load dataset
    print("Loading datasets...")
    train_dataset_raw = load_cnn_dailymail(split='train', num_samples=args.num_train_samples)
    val_dataset_raw = load_cnn_dailymail(split='validation', num_samples=args.num_val_samples)
    
    if train_dataset_raw is None or val_dataset_raw is None:
        print("Failed to load dataset. Exiting.")
        return
        
    # Initialize model
    print(f"Initializing {args.model_name}...")
    summarizer = NewsSummarizer(model_name=args.model_name)
    tokenizer = summarizer.tokenizer
    model = summarizer.model
    device = summarizer.device
    
    # Preprocess dataset
    print("Preprocessing datasets...")
    train_dataset = preprocess_dataset(train_dataset_raw, tokenizer, args.max_source_length, args.max_target_length)
    val_dataset = preprocess_dataset(val_dataset_raw, tokenizer, args.max_source_length, args.max_target_length)
    
    # Prepare DataLoader
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Replace pad tokens with -100 in labels for loss computation
            labels = batch["labels"]
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if step % args.log_every == 0 and step > 0:
                elapsed = time.time() - start_time
                print(f"  Step {step}/{len(train_dataloader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
                
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                labels[labels == tokenizer.pad_token_id] = -100
                
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"  Avg Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  Saving best model to {args.output_dir}")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
        # Generate sample summary
        sample_text = val_dataset_raw[0]['article']
        print(f"\\n  Sample Summary Generation:")
        print(f"  Ref: {val_dataset_raw[0]['highlights'][:150]}...")
        print(f"  Gen: {summarizer.summarize(sample_text)[:150]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train News Summarizer")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Model name")
    parser.add_argument("--num_train_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num_val_samples", type=int, default=500, help="Number of validation samples")
    parser.add_argument("--max_source_length", type=int, default=512, help="Max source length")
    parser.add_argument("--max_target_length", type=int, default=150, help="Max target length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./saved_model", help="Output directory")
    parser.add_argument("--log_every", type=int, default=50, help="Log every X steps")
    
    args = parser.parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        print("Training interrupted.")

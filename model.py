import torch
from typing import Dict, Any, Optional

class NewsSummarizer:
    def __init__(self, model_name: str = 't5-small', device: Optional[str] = None, fine_tuned_path: Optional[str] = None):
        """Load T5 model and tokenizer. Use fine_tuned_path if exists, else load from HuggingFace."""
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        load_path = fine_tuned_path if fine_tuned_path else model_name
        print(f"Loading model from: {load_path} to {self.device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(load_path, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.model.to(self.device)
        self.model_name = model_name
        self.fine_tuned_path = fine_tuned_path

    def summarize(self, text: str, max_input_length: int = 512, max_output_length: int = 150, 
                  min_output_length: int = 30, num_beams: int = 4, length_penalty: float = 2.0, 
                  no_repeat_ngram_size: int = 3, early_stopping: bool = True) -> str:
        """Prepend 'summarize: ' prefix, tokenize, generate with beam search, decode and return string."""
        prefix = "summarize: "
        input_text = prefix + text
        
        inputs = self.tokenizer(input_text, max_length=max_input_length, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_output_length,
                min_length=min_output_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping
            )
            
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return dict with model name, architecture, params, etc."""
        config = self.model.config
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'architecture': config.architectures[0] if hasattr(config, 'architectures') and config.architectures else type(self.model).__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'encoder_layers': config.num_layers,
            'decoder_layers': config.num_decoder_layers,
            'attention_heads': config.num_heads,
            'd_model': config.d_model,
            'vocab_size': config.vocab_size,
            'device': self.device
        }

if __name__ == "__main__":
    print("Testing NewsSummarizer...")
    try:
        summarizer = NewsSummarizer(model_name="t5-small")
        
        sample_text = (
            "Climate change is a long-term shift in global or regional climate patterns. "
            "Often climate change refers specifically to the rise in global temperatures "
            "from the mid-20th century to present. It is primarily caused by human activities "
            "such as the burning of fossil fuels, which releases greenhouse gases like carbon dioxide "
            "and methane into the Earth's atmosphere. These gases trap heat from the sun, leading to a "
            "warming effect known as the greenhouse effect. The consequences of climate change include "
            "more frequent and severe weather events, rising sea levels, and disruptions to ecosystems "
            "and biodiversity. Addressing climate change requires global cooperation and significant "
            "reductions in greenhouse gas emissions."
        )
        
        print("\nModel Info:")
        print(summarizer.get_model_info())
        
        print("\nOriginal Text:")
        print(sample_text)
        
        print("\nSummary:")
        summary = summarizer.summarize(sample_text)
        print(summary)
        
    except Exception as e:
        print(f"Error: {e}")

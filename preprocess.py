import re
import os
import glob
from typing import List, Dict, Any, Tuple, Optional

def clean_text(text: str) -> str:
    """Remove HTML tags, URLs, emails, normalize whitespace, strip special chars (keep punctuation)."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip special chars but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    return text.strip()

def load_cnn_dailymail(split: str = 'train', num_samples: Optional[int] = None) -> Any:
    """Load CNN/DailyMail 3.0.0 via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        try:
            # Try new format first (abisee/cnn_dailymail)
            dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split, trust_remote_code=True)
        except Exception:
            # Fallback to old format
            dataset = load_dataset("cnn_dailymail", "3.0.0", split=split, trust_remote_code=True)
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset
    except ImportError:
        print("Error: HuggingFace 'datasets' library is not installed. Please install it using 'pip install datasets'.")
        return None

def preprocess_dataset(dataset: Any, tokenizer: Any, max_source_length: int = 512, max_target_length: int = 150) -> Any:
    """Tokenize with T5 prefix 'summarize: ', pad to max lengths, return tokenized dataset."""
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")

        labels = tokenizer(text_target=examples["highlights"], max_length=max_target_length, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset

def load_sample_articles(sample_dir: str = 'sample_articles') -> List[Dict[str, str]]:
    """Load .txt files from directory. First line is title. Also load matching reference summary."""
    samples = []
    if not os.path.exists(sample_dir):
        print(f"Directory not found: {sample_dir}")
        return samples
    
    reference_dir = os.path.join(os.path.dirname(sample_dir), 'reference_summaries')
    
    for filepath in glob.glob(os.path.join(sample_dir, '*.txt')):
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                continue
            title = lines[0].strip()
            if title.startswith('# '):
                title = title[2:]
            content = "".join(lines[1:]).strip()
            
        reference_path = os.path.join(reference_dir, filename)
        reference = ""
        if os.path.exists(reference_path):
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference = f.read().strip()
                
        samples.append({
            'filename': filename,
            'title': title,
            'content': content,
            'reference': reference
        })
        
    return samples

def extract_text_from_file(uploaded_file: str) -> str:
    """Extract text from .txt, .pdf, .docx files."""
    if not os.path.exists(uploaded_file):
        raise FileNotFoundError(f"File not found: {uploaded_file}")
        
    ext = os.path.splitext(uploaded_file)[1].lower()
    
    if ext == '.txt':
        with open(uploaded_file, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        try:
            import PyPDF2
            text = ""
            with open(uploaded_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + " "
            return text.strip()
        except ImportError:
            print("Error: PyPDF2 is not installed. Please install it using 'pip install PyPDF2'.")
            return ""
    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            print("Error: python-docx is not installed. Please install it using 'pip install python-docx'.")
            return ""
    else:
        raise ValueError(f"Unsupported file format: {ext}")

if __name__ == "__main__":
    print("Testing clean_text...")
    raw_text = "This is an <a href='http://example.com'>example</a> text! Contact me at test@email.com.    Lots of spaces here. #123 $$"
    print(f"Raw: {raw_text}")
    print(f"Cleaned: {clean_text(raw_text)}")
    
    print("\nTesting load_cnn_dailymail...")
    ds = load_cnn_dailymail(split='train', num_samples=2)
    if ds:
        print(f"Loaded {len(ds)} samples.")
        print("Sample article snippet:")
        print(ds[0]['article'][:100] + "...")
        print("Sample highlight snippet:")
        print(ds[0]['highlights'][:100] + "...")

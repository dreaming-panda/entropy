import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
check_min_version("4.28.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")


def convert_wiki_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:400]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_cnn_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:400]")
    def tokenize_function(examples):
            return tokenizer(examples["article"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

# Imports
from tokenizers import ByteLevelBPETokenizer
from transformers import BartConfig, BartTokenizerFast, AutoModelForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import pipeline, Trainer, TrainingArguments


class NovelEncoder():
    def __init__(self) -> None:
        # Use the Shakespere dataset.
        paths = ['/kaggle/input/shakespeare-all/shakespeare.txt']

        # byte-level byte-pair encoding tokenizer 
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=paths, vocab_size=50000, min_frequency=2, special_tokens=[
            "<s>",
            "</s>",
            "<pad>"
        ])

        # Save the tokenizer model in a folder(contains 2 files - vocab.json and mergest.txt) 
        !mkdir shakespere_BART
        tokenizer.save_model("shakespere_BART")
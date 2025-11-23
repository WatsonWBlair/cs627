from datasets import load_dataset, load_from_disk
from Encoders.text.SEED import SEED

# Loads dataset from local resources.
def load_local(path: str):
    local_data = load_from_disk(path)
    return local_data

# downloads dataset from huggingface.
def load_remote_MTEB():
    try:
        sts_dataset = load_dataset("mteb/stsbenchmark-sts", split="train") 
        print(f"\nSuccessfully loaded 'mteb/stsbenchmark-sts' (English) Test set. Number of examples: {len(sts_dataset)}")
    except Exception as e:
        print(f"\nCould not load mteb/stsbenchmark-sts dataset: {e}")

    encoder = SEED()
    def encode(examples):
        examples['sentence2'] = encoder.batch(examples['sentence2'])
        return examples 

    encoded_data = sts_dataset.map(encode, batched=True)

    encoded_data.save_to_disk('SEED_ENC_MTEB_DATASET')

    return encoded_data


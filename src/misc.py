from datasets import load_dataset
from datasets import load_from_disk

# Dataset dwonload
# print("Downloading dataset...")
# dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
# save_path = "./wikitext_103_raw"
# dataset.save_to_disk(save_path)
# print(f"Success! Dataset saved to: {save_path}")


# Load dataset
dataset = load_from_disk("../wikitext_103_raw")
print("Loaded from disk!")
print(dataset)

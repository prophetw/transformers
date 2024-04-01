from datasets import load_dataset, Audio

# NLP natural language processing
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)

# audio
dataset = load_dataset("PolyAI/minds14", name="zh-CN", split="train")

# computer vision
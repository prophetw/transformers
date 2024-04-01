from datasets import load_dataset, Audio

# NLP natural language processing
from transformers import AutoTokenizer, pipeline

# classifier = pipeline('sentiment-analysis')
# classifier("We are very happy to show you the 🤗 Transformers library.")

# dataset = load_dataset("yelp_review_full")
# print(dataset["train"][100])

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
# encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
# print(encoded_input)

# audio
dataset = load_dataset(
	"PolyAI/minds14", 
	"zh-CN",
	split="train",
	trust_remote_code=True
	)
print(dataset)

# computer vision
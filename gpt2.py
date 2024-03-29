from transformers import GPT2Model, GPT2Tokenizer

model_load_path = "./modelData/gpt2-model"
tokenizer_load_path = "./modelData/gpt2-tokenizer"

model = GPT2Model.from_pretrained(model_load_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_load_path)
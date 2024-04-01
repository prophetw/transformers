from transformers import GPT2Model, GPT2Tokenizer

model_name = "gpt2"  # model name from huggingface
model = GPT2Model.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model_save_path = "./modelData/gpt2-model"
tokenizer_save_path = "./modelData/gpt2-tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)
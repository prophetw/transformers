from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # 举例使用GPT-2模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 示例：生成文本
input_ids = tokenizer.encode("const bubbleSort = (numAry: number[]) => {", return_tensors="pt")
generated_ids = model.generate(input_ids, max_length=50)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

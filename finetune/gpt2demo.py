from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


train_path = 'path_to_train.txt'
validation_path = 'path_to_validation.txt'

# 使用TextDataset准备数据集
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128)

validation_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=validation_path,
    block_size=128)

# 数据整理，主要用于将数据批处理和准备输入格式
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

trainer.train()

# if you want to save the model
# 完成训练后保存模型
# trainer.save_model("path_to_save_your_finetuned_model")

# 你也可以选择保存分词器，以确保后续加载模型时使用相同的分词逻辑 建议使用这个 保证一样的输入
# tokenizer.save_pretrained("path_to_save_your_finetuned_model")
# model = GPT2LMHeadModel.from_pretrained("path_to_save_your_finetuned_model")
# tokenizer = GPT2Tokenizer.from_pretrained("path_to_save_your_finetuned_model")



prompt = "今天天气如何？"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# 生成文本
output = model.generate(input_ids, max_length=50, num_return_sequences=5)
print("Generated text:\n" + 100 * '-')
for i, output_ids in enumerate(output):
    print(f"{i}: {tokenizer.decode(output_ids, skip_special_tokens=True)}")

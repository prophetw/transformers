from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import load_dataset

# 加载和预处理数据
dataset = load_dataset("text", data_files={"train": "path_to_train.txt", "validation": "path_to_validation.txt"})
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义模型配置和初始化模型
config = DistilBertConfig(num_labels=2)
model = DistilBertForSequenceClassification(config=config)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# 训练模型
trainer.train()

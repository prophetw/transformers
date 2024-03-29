
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def main():
    # 加载模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained("./modelData/gpt2-tokenizer")
    model = GPT2LMHeadModel.from_pretrained("./modelData/gpt2-model")

    # 设置模型为评估模式，关闭dropout
    model.eval()

    # 开始交互式会话
    print("GPT-2 交互式问答模式。输入'exit'来结束会话。")
    while True:
        # 获取用户输入
        input_text = input("用户: ")
        if input_text == "exit":
            break
        
        # 编码用户输入并添加终止符
        input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

        # 生成响应
        # 注意: 你可以通过调整max_length参数来限制响应长度
        output = model.generate(
            input_ids, 
            max_length=200, 
            temperature=0.7,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id)

        # 解码并打印模型响应
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("GPT-2: ", response_text)

if __name__ == "__main__":
    main()

import whisper
import pprint
from datasets import load_dataset, Audio
# 加载模型
# https://github.com/openai/whisper
# tiny base small medium large 
model = whisper.load_model("small")


# dataset = load_dataset(
# 	"PolyAI/minds14", 
# 	"zh-CN",
# 	split="train",
# 	trust_remote_code=True
# )

# 语音文件路径
# audioPath = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~BALANCE/603525386b30c74e5584cd24.wav"
# audioPath = 
# audioPath = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~BALANCE/603525386b30c74e5584cd24.wav"

test_path = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~BALANCE/603525386b30c74e5584cd24.wav"
#  'transcription': '我的账户还有多少钱呢'

# 'sampling_rate': 8000},
# test_path = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~BALANCE/60352e526b30c74e5584ce3f.wav"
#  'transcription': '我想要查询我的账户余额'}

# 你好我想要更改我的地址
# test_path = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~ADDRESS/603520926b30c74e5584cc44.wav"



# pprint.pprint(dataset[2])
# audioPath = dataset[2]["path"]
# print(audioPath)

# 处理语音文件并进行识别
result = model.transcribe(test_path)

# 打印识别结果
print(result["text"])

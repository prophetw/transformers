import whisper
import pprint
from datasets import load_dataset, Audio
# 加载模型
model = whisper.load_model("tiny")


dataset = load_dataset(
	"PolyAI/minds14", 
	"zh-CN",
	split="train",
	trust_remote_code=True
)

pprint.pprint(dataset[2])
audioPath = dataset[2]["path"]

# 处理语音文件并进行识别
result = model.transcribe(audioPath)

# 打印识别结果
print(result["text"])

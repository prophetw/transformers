
from datasets import load_dataset
import pprint

# audio
dataset = load_dataset(
	"PolyAI/minds14", 
	"zh-CN",
	split="train",
	trust_remote_code=True
)
data = dataset[350]
sample = data["audio"]
audio_sample_rate = sample["sampling_rate"]
audio_path = sample["path"]
pprint.pprint(dataset)
pprint.pprint(data)
# pprint.pprint(sample)
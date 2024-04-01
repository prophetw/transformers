from datasets import load_dataset, Audio
import soundfile as sf
import pprint
import librosa
# NLP natural language processing
from transformers import AutoTokenizer, pipeline, WhisperProcessor, WhisperForConditionalGeneration

# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")


processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# audio
dataset = load_dataset(
	"PolyAI/minds14", 
	"zh-CN",
	split="train",
	trust_remote_code=True
	)

sample = dataset[2]["audio"]
# print(dataset)
audio_sample_rate = sample["sampling_rate"]
audio_path = sample["path"]
pprint.pprint(dataset[2])


speech, sample_rate = librosa.load(audio_path, sr=8000)

# print(sample_rate)
# 然后，将音频从8000Hz重采样到16000Hz
speech_resampled = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

input_features = processor(speech_resampled, sampling_rate=16000, return_tensors="pt").input_features 
# pprint.pprint(input_features)



predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
pprint.pprint(transcription)


# computer vision

# 加载音频文件
# speech, sample_rate = sf.read(audio_path)

# 假设 `speech` 是你的原始音频数据，`sample_rate` 是原始采样率（在这个例子中是8000Hz）
# 首先，读取你的音频文件
# speech, sample_rate = librosa.load(audio_path, sr=8000)

# print(sample_rate)
# 然后，将音频从8000Hz重采样到16000Hz
# speech_resampled = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
# pprint.pprint(speech_resampled)

# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

from faster_whisper import WhisperModel


# https://github.com/systran/faster-whisper

# model_size = "large-v3"
# model_size = "large-v3"
# model_size = "large-v3"
# model_size = "large-v2"
model_size = "small"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")



test_path = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~BALANCE/603525386b30c74e5584cd24.wav"
#  'transcription': '我的账户还有多少钱呢'

# 'sampling_rate': 8000},
# test_path = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~BALANCE/60352e526b30c74e5584ce3f.wav"
#  'transcription': '我想要查询我的账户余额'}

# 你好我想要更改我的地址
# test_path = "/root/.cache/huggingface/datasets/downloads/extracted/3e8fd2b4a184c2bec90357f47da393914d54ba89c7f887733ff8e2705be8f6fe/zh-CN~ADDRESS/603520926b30c74e5584cc44.wav"

segments, info = model.transcribe(test_path, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
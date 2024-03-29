from huggingface_hub import hf_hub_download


# hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./modelData/bigscience_t0")
hf_hub_download(repo_id="openai-community/gpt2", filename="config.json", cache_dir="./modelData/openai_gpt2")


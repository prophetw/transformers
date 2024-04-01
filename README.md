# [huggingface.co](https://huggingface.co/)

* ubuntu 
* Python 3.6+
* PyTorch 1.1.0+
* VPN needed or use Mirror

## mirror
export HF_ENDPOINT=https://hf-mirror.com

## dependicies 
```bash
# soundfile
apt install libsndfile1 libffi-dev

# python3 
python3 -m venv .env
# active
source .env/bin/activate

# generate requirements.txt
pip freeze > requirements.txt

# install dep
pip install -r requirements.txt

# check your internet connection maybe you need to set proxy
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"

```

# Offline environment
```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```
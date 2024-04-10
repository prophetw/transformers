# [huggingface.co](https://huggingface.co/)

* ubuntu 
* Python 3.6+
* PyTorch 1.1.0+
* VPN needed or use Mirror https://hf-mirror.com/ 

## mirror 
> mirror for huggingface
in bash 
export HF_ENDPOINT=https://hf-mirror.com

vi ~/.bashrc  ~/.zshrc
export HF_ENDPOINT=https://hf-mirror.com

> mirror for pip
pip install soundfile -i https://pypi.tuna.tsinghua.edu.cn/simple/
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
vi ~/.bashrc  ~/.zshrc
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

## dependicies 
```bash
# soundfile
apt install libsndfile1 libffi-dev

# pip 镜像地址
pip install soundfile -i https://pypi.tuna.tsinghua.edu.cn/simple/

# python3 
python3 -m venv .venv
# active
source .venv/bin/activate

# generate requirements.txt
pip freeze > requirements.txt

# install dep
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# check your internet connection maybe you need to set proxy
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"

```

# Offline environment
```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```
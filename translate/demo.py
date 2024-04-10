# translate.py
from transformers import pipeline

def translate(text, model_name):
    translator = pipeline("translation", model=model_name)
    translation = translator(text)[0]['translation_text']
    return translation

if __name__ == "__main__":
    import sys
    text = sys.argv[1]
    # if not exist text 

    model_name = sys.argv[2]
    print(translate(text, model_name))
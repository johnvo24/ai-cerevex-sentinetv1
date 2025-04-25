import yaml
from transformers import BertTokenizer

with open('config.yaml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model_name = config['finetune']['name']

class Preprocessing:
    def __init__(self, input=None, max_length=None):
        self.input = input
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(self):
        return self.tokenizer(self.input['text'], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

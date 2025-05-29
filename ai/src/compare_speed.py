import time
import yaml
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

from first_phase.preprocessing import Preprocessing
from second_phase.reinforcement_learning import ActorCritic
    
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
config = load_config()
SFT_model_path = config['rl_training']['SFT_model_path']
max_length = config['input']['max_seq_length']
num_classes = config['input']['num_classes']

model = BertForSequenceClassification.from_pretrained(SFT_model_path, num_labels=num_classes)
tokenizer = BertTokenizer.from_pretrained(SFT_model_path)

actor_critic = ActorCritic(model)
actor_critic.load_state_dict(torch.load('model/actor_critic.pt'))

def mapping_label(label):
    if label == 0:
        return 'World'
    elif label == 1:
        return 'Sports'
    elif label == 2:
        return 'Business'
    elif label == 3:
        return 'Sci/Tech'

def use_first_phase(text):
    start = time.time()

    preprocessor = Preprocessing(text, max_length=max_length)
    predict = model(**preprocessor.tokenize())
    predict = nn.Softmax(dim=-1)(predict.logits)
    predict = torch.argmax(predict, dim=-1)
    print(mapping_label(predict.item()))

    end = time.time()
    elapsed = end - start

    return elapsed

def use_second_phase(text):
    start = time.time()

    preprocessor = Preprocessing(text, max_length=max_length)
    input_ids = preprocessor.tokenize()['input_ids']
    for input_id in input_ids:
        i = 2
        chunk_ids = input_id[0:2]
        action_prob, _ = actor_critic(chunk_ids.unsqueeze(0))
        action = torch.argmax(action_prob, dim=-1)
        while(action.item() != 1 and i < len(input_id) - 2):
            chunk_ids = torch.cat((chunk_ids, input_id[i:i+2]), dim=0)
            i += 1
            action_prob, _ = actor_critic(chunk_ids.unsqueeze(0))
            action = torch.argmax(action_prob, dim=-1)

        predict = model(chunk_ids.unsqueeze(0))
        predict = nn.Softmax(dim=-1)(predict.logits)
        predict = torch.argmax(predict, dim=-1)
        print(mapping_label(predict.item()))

    end = time.time()
    elapsed = end - start

    return elapsed

input = {'text': 'xin chào tất cả mọi người'}
t1 = use_first_phase(input)
t2 = use_second_phase(input)

print(t1)
print(t2)

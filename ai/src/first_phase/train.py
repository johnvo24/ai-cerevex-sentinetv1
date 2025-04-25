import yaml
from datasets import load_dataset
from preprocessing import Preprocessing, AGNewsDataset
from transformers import BertForSequenceClassification, TrainingArguments, Trainer, BertTokenizer
import torch

# Load config
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
config = load_config()
max_seq_length = config['input']['max_seq_length']
num_classes = config['input']['num_classes']
model_name = config['finetune']['name']

# Load dataset
dataset = load_dataset(config['data']['ag_news'])
train_data = dataset['train'][:32]
test_data = dataset['test'][:32]

# Preprocess dataset
train_preprocessor = Preprocessing(train_data, max_seq_length)
test_preprocessor = Preprocessing(test_data, max_seq_length)
train_vectors = train_preprocessor.tokenize()
test_vectors = test_preprocessor.tokenize()

train_data = AGNewsDataset(train_vectors['input_ids'], train_vectors['attention_mask'], train_data['label'])
test_data = AGNewsDataset(test_vectors['input_ids'], test_vectors['attention_mask'], test_data['label'])

# Supervised Finetuning BERT on AG News
bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
tokenizer = BertTokenizer.from_pretrained(model_name)

# finetune configs
training_args = TrainingArguments(
    output_dir=config["finetune"]["save_model_path"],
    num_train_epochs=config['finetune']['epochs'],
    per_device_train_batch_size=config['finetune']['batch_size'],
    per_device_eval_batch_size=config['finetune']['batch_size'],
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=config["logging"]["log_dir"],
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch"
)

# Training
trainer = Trainer(
    model=bert,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer
)

trainer.train()

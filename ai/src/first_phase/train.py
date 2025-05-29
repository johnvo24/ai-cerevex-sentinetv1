import yaml
from datasets import load_dataset
from preprocessing import Preprocessing, AGNewsDataset
from transformers import BertForSequenceClassification, TrainingArguments, Trainer, BertTokenizer
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use")

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
print(f'Load dataset...')
dataset = load_dataset(config['data']['ag_news'])
train_data = dataset['train']
test_data = dataset['test']

# Preprocess dataset
print(f'Process dataset...')
train_preprocessor = Preprocessing(train_data, max_seq_length)
test_preprocessor = Preprocessing(test_data, max_seq_length)
train_vectors = train_preprocessor.tokenize()
test_vectors = test_preprocessor.tokenize()

train_data = AGNewsDataset(train_vectors['input_ids'], train_vectors['attention_mask'], train_data['label'])
test_data = AGNewsDataset(test_vectors['input_ids'], test_vectors['attention_mask'], test_data['label'])

# Supervised Finetuning BERT on AG News
bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
tokenizer = BertTokenizer.from_pretrained(model_name)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# finetune configs
# training_args = TrainingArguments(
#     output_dir=config["finetune"]["save_model_path"],
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir=config["logging"]["log_dir"],
#     logging_steps=1,
#     eval_strategy="epoch",
#     save_strategy="epoch"
# )

training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",    
    greater_is_better=True,
    num_train_epochs=config['finetune']['epochs'],
    per_device_train_batch_size=config['finetune']['batch_size'],
    per_device_eval_batch_size=config['finetune']['batch_size']*2,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=config['finetune']['learning_rate'],
    gradient_accumulation_steps=1,
    fp16=True,
    report_to="none"
)

# Training
trainer = Trainer(
    model=bert,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

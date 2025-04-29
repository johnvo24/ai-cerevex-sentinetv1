from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
import os
import yaml

# Load config
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
config = load_config()
dataset = load_dataset(config['data']['ag_news'])
model_name = config['finetune']['name']
max_len = config['input']['max_seq_length']

# Load AG News dataset
print("Loading dataset...")
dataset = load_dataset(config['data']['ag_news'])
train_data = dataset['train']
test_data = dataset['test']
train_labels = [example['label'] for example in train_data]
test_labels = [example['label'] for example in test_data]
label_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

train_label_counts = Counter(train_labels)
test_label_counts = Counter(test_labels)

x_train = [label_names[label] for label in train_label_counts.keys()]
y_train = [count for count in train_label_counts.values()]
x_test = [label_names[label] for label in test_label_counts.keys()]
y_test = [count for count in test_label_counts.values()]

train_lengths = [len(example['text'].split()) for example in train_data]
test_lengths = [len(example['text'].split()) for example in test_data]

def distribution_plot(x_train, y_train, x_test, y_test):
    plt.figure(figsize=(8,6))
    plt.bar(x_train, y_train)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Label Distribution in AG News Training Set')
    plt.grid(axis='y')

    output_dir = "res"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ag_news_train_label_distribution.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()


    plt.figure(figsize=(8,6))
    plt.bar(x_test, y_test)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Label Distribution in AG News Test Set')
    plt.grid(axis='y')

    output_dir = "res"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ag_news_test_label_distribution.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def length_histogram(train_lengths, test_lengths):
    output_dir = "res"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.hist(train_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Text Length (words)')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Text Lengths in AG News Training Set')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, "train_length_histogram.png"))
    plt.close()
    print("Saved text length histogram.")

    plt.figure(figsize=(8,6))
    plt.hist(test_lengths, bins=30, color='lightgreen', edgecolor='black')
    plt.xlabel('Text Length (words)')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Text Lengths in AG News Test Set')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, "test_length_histogram.png"))
    plt.close()
    print("Saved text length histogram.")

def length_boxplot(train_lengths, test_lengths):
        output_dir = "res"
        plt.figure(figsize=(8,6))
        plt.boxplot([train_lengths, test_lengths], labels=['Train', 'Test'], vert=False)
        plt.xlabel('Text Length (words)')
        plt.title('Boxplot of Text Lengths in AG News')
        plt.grid(axis='x')
        plt.savefig(os.path.join(output_dir, "text_length_boxplot.png"))
        plt.close()
        print("Saved title length boxplot.")

def embedding_space(data, title, output_dir):
    from transformers import BertTokenizer, BertModel

    print("Loading model, tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    pca = PCA(n_components=2)
    model.eval()

    print("Generating embeddings...")
    vectors = tokenizer(data['text'], padding=True, truncation=True, max_length=max_len, return_tensors='pt')

    cls_embs = []
    labels = []

    print("Generating embedding space...")
    for sample in vectors:
        with torch.no_grad():
            output = model(sample['input_ids'], sample['attention_mask'])
            cls_emb = output.last_hidden_state[:, 0, :]
            cls_embs.append(cls_emb)
            labels.append(sample['label'])

    cls_embs = np.array(cls_embs)
    labels = np.array(labels)

    emb_2d = pca.fit_transform(cls_embs)

    print('Plotting...')
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    colors = ["red", "blue", "green", "orange"]

    plt.figure(figsize=(10,8))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], c=colors[label], label=label_names[label], alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    plt.savefig(output_dir)
    plt.close()
    print("Saved 2D embedding plot!")

# distribution_plot(x_train, y_train, x_test, y_test)
# length_histogram(train_lengths, test_lengths)
# length_boxplot(train_lengths, test_lengths)
embedding_space(train_data, 'BERT Embedding Space of AG News Training Set', 'res/ag_news_train_embedding_space.png')
embedding_space(test_data, 'BERT Embedding Space of AG News Test Set', 'res/ag_news_test_embedding_space.png')

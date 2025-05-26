import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification
from environment import TextEnv, split_dataset
from torch.optim import AdamW
from datasets import load_dataset
from reinforcement_learning import PPO, ActorCritic
import yaml
from tqdm import tqdm

# Load config
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
config = load_config()
episode = config['rl_training']['episode']
learning_rate = config['rl_training']['learning_rate']
num_data = config['rl_training']['num_data']
clip_ratio = config['rl_training']['clip_ratio']
SFT_model_path = config['rl_training']['SFT_model_path']
k = config['rl_training']['k']
    
# Load SFT model
print('Loading SFT model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(SFT_model_path, num_labels=4)
tokenizer = BertTokenizer.from_pretrained(SFT_model_path)

# Load dataset
print('Loading dataset...')
dataset = load_dataset(config['data']['ag_news'])
print('Tokenizing dataset...')
train_data = split_dataset(dataset['train'], samples_per_class=num_data)
train_token = tokenizer(train_data['text'], padding=False, return_tensors=None)
train_token = [torch.tensor(token).to(device) for token in train_token['input_ids']]
train_label = torch.tensor(train_data['label'])
train_label = train_label.to(device)
# test_data = dataset['test']
del dataset
del train_data

# Init RL
actor_critic = ActorCritic(model)
actor_critic.to(device)

# for param in actor_critic.model.parameters():
#     param.requires_grad = False

optimizer = AdamW(actor_critic.parameters(), lr=learning_rate)
ppo = PPO(actor_critic, optimizer, clip_ratio)
env = TextEnv(train_token, train_label, model, k, device)

last_hidden_lock = True

# Start training
print('Start training...')
process = tqdm(range(episode))
for ep in process:
    if ep >= episode // 2 and last_hidden_lock:
        last_hidden_lock = False
        for name, param in actor_critic.model.named_parameters():
            if 'encoder.layer.11' in name or 'pooler' in name:
                param.requires_grad = True
        print('Last hidden layer unlocked')

    state = env.reset()
    avg_reward = 0
    avg_loss = 0
    count = 0

    for sentence in range(num_data*4):
        done = False
        total_reward = 0
        total_critic_loss = 0
        loss_count = 0

        while not done:
            state = state.to(device)
            action_probs, _ = actor_critic(state) 
            action = torch.multinomial(action_probs, 1)

            next_state, reward, done = env.step(action)
            total_reward += reward

            loss = ppo.update(state, action, reward, next_state)
            total_critic_loss += loss
            loss_count += 1
            state = next_state

        count += 1
        avg_reward += total_reward
        avg_loss += total_critic_loss / loss_count
        state = env.next_sentence()
        process.set_description(f"Episode {ep+1}/{episode}, Data no.{sentence+1}/{num_data*4}, Avg Reward: {avg_reward/count:.2f}")

        if (sentence+1) % 100 == 0:
            tqdm.write(f'Episode {ep+1}/{episode}, Avg Reward: {avg_reward/count:.2f}, Avg Critic Loss: {avg_loss/loss_count:.2f}')

# Save policy model
print('Saving policy model...')
torch.save(actor_critic.state_dict(), 'model/actor_critic.pt')

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

optimizer = AdamW(actor_critic.parameters(), lr=learning_rate)
ppo = PPO(actor_critic, optimizer, clip_ratio)
env = TextEnv(train_token, train_label, model, k, device)

# Start training
print('Start training...')
best_actorcritic = actor_critic
best_avg_reward = -999.0
best_epoch = 0
process = tqdm(range(episode))
for ep in process:
    state = env.reset()
    avg_reward = 0
    count = 0

    for sentence in range(num_data*4):
        done = False
        total_reward = 0

        while not done:
            state = state.to(device)
            action_probs, _ = actor_critic(state)
            action = torch.multinomial(action_probs, 1)

            next_state, reward, done = env.step(action)
            total_reward += reward

            ppo.update(state, action, total_reward, next_state)
            state = next_state

        count += 1
        avg_reward += total_reward
        state = env.next_sentence()
        process.set_description(f"Episode {ep+1}/{episode}, Data no.{sentence+1}/{num_data*4}, Avg Reward: {avg_reward/count:.2f}")
    
    if avg_reward > best_avg_reward:
        best_actorcritic = actor_critic
        best_avg_reward = avg_reward
        best_epoch = ep

# Save policy model
print('Saving policy model...')
torch.save(best_actorcritic.state_dict(), f'model/best_actor_critic_{best_epoch}.pt')

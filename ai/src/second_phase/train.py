import torch
from transformers import BertTokenizer, BertForSequenceClassification
from environment import TextEnv
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
    
# Load dataset
print('Loading dataset...')
dataset = load_dataset(config['data']['ag_news'])
train_data = dataset['train'][:32]
test_data = dataset['test'][:32]
print(train_data)

# Load SFT model
print('Loading SFT model...')
model = BertForSequenceClassification.from_pretrained('model/checkpoint-3', num_labels=4)
tokenizer = BertTokenizer.from_pretrained('model/checkpoint-3')

# Init RL
optimizer = AdamW(model.parameters(), lr=5e-5)
actor_critic = ActorCritic(model)
ppo = PPO(actor_critic, optimizer)
env = TextEnv(train_data, tokenizer, max_seq_length=128)

# Start training
print('Start training...')
for ep in tqdm(range(episode), desc='Episode'):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action_probs, _ = actor_critic(state) 
        action = torch.multinomial(action_probs, 1)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        ppo.update(state, action, reward, next_state)
        state = next_state

    tqdm.write(f"Episode {episode}, Total Reward: {total_reward}")  # Show episode result

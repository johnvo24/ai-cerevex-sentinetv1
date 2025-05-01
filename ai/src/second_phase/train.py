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
learning_rate = config['rl_training']['learning_rate']
batch_size = config['rl_training']['batch_size']
clip_ratio = config['rl_training']['clip_ratio']
SFT_model_path = config['rl_training']['SFT_model_path']
    
# Load dataset
print('Loading dataset...')
dataset = load_dataset(config['data']['ag_news'])
train_data = dataset['train']
test_data = dataset['test']

# Load SFT model
print('Loading SFT model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(SFT_model_path, num_labels=4)
tokenizer = BertTokenizer.from_pretrained(SFT_model_path)

# Init RL
actor_critic = ActorCritic(model)
actor_critic.to(device)
optimizer = AdamW(actor_critic.parameters(), lr=learning_rate)
ppo = PPO(actor_critic, optimizer, clip_ratio)
env = TextEnv(train_data, model, tokenizer, device)

# Start training
print('Start training...')
for ep in tqdm(range(episode), desc='Episode'):
    state = env.reset()

    for sentence in range(batch_size*4):
        done = False
        total_reward = 0

        while not done:
            state = state.to(device)
            action_probs, _ = actor_critic(state) 
            action = torch.multinomial(action_probs, 1)

            next_state, reward, done = env.step(action)
            total_reward += reward

            ppo.update(state, action, reward, next_state)
            state = next_state

        state = env.next_sentence()
        tqdm.write(f"Episode {ep+1}/{episode}, Batch {sentence+1}/{batch_size*4}, Total Reward: {total_reward}")  # Show episode result

# Save policy model
print('Saving policy model...')
torch.save(actor_critic.state_dict(), 'model/actor_critic.pt')

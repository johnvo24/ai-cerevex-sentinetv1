import gym
import torch
import torch.nn as nn
import yaml
import random
from collections import defaultdict, Counter


# Load config
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
config = load_config()
action_read = config['rl_training']['action_read']
action_predict = config['rl_training']['action_predict']
reward_predict = config['rl_training']['reward_predict']
reward_predict_at_the_end = config['rl_training']['reward_predict_at_the_end']
penalty_predict = config['rl_training']['penalty_predict']
penalty_predict_at_the_end = config['rl_training']['penalty_predict_at_the_end']
penalty_read = config['rl_training']['penalty_read']
num_data = config['rl_training']['num_data']

def split_dataset(dataset, samples_per_class=1250):
    # Group samples by label
    label_groups = defaultdict(list)
    for i, example in enumerate(dataset):
        label_groups[example['label']].append(i)

    # Sample equal number from each class
    balanced_indices = []

    for label, indices in label_groups.items():
        sampled = random.sample(indices, samples_per_class)
        balanced_indices.extend(sampled)

    # Final balanced subset
    balanced_subset = dataset.select(balanced_indices)

    # check
    labels = [ex["label"] for ex in balanced_subset]

    return balanced_subset

class TextEnv(gym.Env):
    def __init__(self, tokens, labels, model, k, device):
        self.tokens = tokens
        self.labels = labels
        self.model = model
        self.current_tokens_index = 0
        self.current_token_index = 0
        self.k = k
        self.token_chunk = [self.tokens[self.current_tokens_index][self.current_token_index]]
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def reset(self):
        self.current_tokens_index = 0
        self.current_token_index = 0
        self.token_chunk = [self.tokens[self.current_tokens_index][self.current_token_index]]
        return self._get_state(self.token_chunk)

    def next_sentence(self):
        self.current_tokens_index += 1
        self.current_token_index = 0

        if self.current_tokens_index == len(self.tokens):
            return None

        self.token_chunk = [self.tokens[self.current_tokens_index][self.current_token_index]]
        return self._get_state(self.token_chunk)

    def _get_state(self, chunk):
        if isinstance(chunk, torch.Tensor):
            return chunk.unsqueeze(0).to(self.device)
        else:
            return torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(self.device)

    def step(self, action):
        chunk = self.tokens[self.current_tokens_index]
        max_len = len(chunk)
        true_action = self.labels[self.current_tokens_index]

        # Continue Reading
        if action == action_read:
            if len(self.token_chunk) < max_len: # Continue read
                reward = penalty_read
                self.current_token_index = min(self.current_token_index + self.k, max_len - 1)
                self.token_chunk = chunk[:self.current_token_index + 1]
                done = False

            else: # Force predict, reach the end
                predict = self.model(self._get_state(self.token_chunk))
                predict = self.softmax(predict.logits)
                label = torch.argmax(predict, dim=-1)

                if label == true_action:
                    reward = reward_predict_at_the_end
                else:
                    reward = penalty_predict_at_the_end
                done = True

        # Predict action
        else:
            predict = self.model(self._get_state(self.token_chunk))
            predict = self.softmax(predict.logits)
            label = torch.argmax(predict, dim=-1)

            if max_len > len(self.token_chunk): # Early predict
                if label == true_action:
                    reward = reward_predict
                else:
                    reward = penalty_predict
            else: # Reach the end
                if label == true_action:
                    reward = reward_predict_at_the_end
                else:
                    reward = penalty_predict_at_the_end
            done = True

        next_state = self._get_state(self.token_chunk)
        return next_state, reward, done

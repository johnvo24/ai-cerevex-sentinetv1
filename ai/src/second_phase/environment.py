import gym
import torch
import torch.nn as nn
import random
from collections import defaultdict
import second_phase.config as configs

# Load config
config = configs.load_config()
action_read = config['rl_training']['action_read']
action_predict = config['rl_training']['action_predict']
reward_predict = config['rl_training']['reward_predict']
reward_predict_at_the_end = config['rl_training']['reward_predict_at_the_end']
penalty_predict = config['rl_training']['penalty_predict']
penalty_predict_at_the_end = config['rl_training']['penalty_predict_at_the_end']
penalty_read = config['rl_training']['penalty_read']

class TextEnv(gym.Env):
    def __init__(self, tokens, labels, model, k, device):
        self.tokens = tokens
        self.labels = labels
        self.model = model
        self.k = k
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.reset()

    def reset(self):
        self.current_tokens_index = 0
        self.current_token_index = self.k - 1
        self.token_chunk = self.tokens[self.current_tokens_index][: self.current_token_index + 1]
        return self._get_state()

    def next_sentence(self):
        self.current_tokens_index += 1
        self.current_token_index = self.k - 1

        if self.current_tokens_index == len(self.tokens): return None

        self.token_chunk = self.tokens[self.current_tokens_index][: self.current_token_index + 1]
        return self._get_state()

    def _get_state(self):
        return self.token_chunk.unsqueeze(0).to(self.device) if isinstance(self.token_chunk, torch.Tensor) else torch.tensor(self.token_chunk, dtype=torch.long).unsqueeze(0).to(self.device)

    def _predict_label(self):
        logits = self.model(self._get_state()).logits
        return torch.argmax(self.softmax(logits), dim=-1)

    def step(self, action, pad_token_id):
        chunk = self.tokens[self.current_tokens_index]
        true_label = self.labels[self.current_tokens_index]
        max_len = len(chunk)

        # Continue Reading
        next_token_index = min(self.current_token_index + self.k, max_len - 1)
        is_next_k_not_full_token_id = bool((chunk[self.current_token_index+1:next_token_index+1] != pad_token_id).any().item())
        if action == action_read:
            if len(self.token_chunk) < max_len and is_next_k_not_full_token_id == True: # Continue read
                self.current_token_index = next_token_index
                self.token_chunk = chunk[:self.current_token_index + 1]
                reward, done = penalty_read, False
            else: # Force predict, reach the end
                pred_label = self._predict_label()
                reward = reward_predict_at_the_end if pred_label == true_label else penalty_predict_at_the_end
                done = True

        # Predict action
        else:
            pred_label = self._predict_label()
            if len(self.token_chunk) < max_len:
                reward = reward_predict if pred_label == true_label else penalty_predict
            else:
                reward = reward_predict_at_the_end if pred_label == true_label else penalty_predict_at_the_end
            done = True

        next_state = self._get_state()
        return next_state, reward, done
    

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
    return balanced_subset

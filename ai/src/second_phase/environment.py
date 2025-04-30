import gym
import torch
import torch.nn as nn
import yaml


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

class TextEnv(gym.Env):
    def __init__(self, dataset, model, tokenizer):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.current_dataset_index = 0
        self.current_sentence_index = 0
        self.word_chunk = [self.dataset['text'][self.current_dataset_index].split(' ')[self.current_sentence_index]]
        self.softmax = nn.Softmax(dim=1)
        # self.action_space = gym.spaces.Discrete(2)
        # self.observation_space = gym.spaces.Discrete(self.max_seq_length)

    def reset(self):
        self.current_dataset_index = 0
        self.current_sentence_index = 0
        self.word_chunk = [self.dataset['text'][self.current_dataset_index].split(' ')[self.current_sentence_index]]
        return self._get_state(self.word_chunk)

    def _get_state(self, word_chunk):
        text = ' '.join(word_chunk)
        encoded = self.tokenizer(text, padding=False, return_tensors='pt')
        return encoded['input_ids']

    def step(self, action):
        max_len = len(self.dataset['text'][self.current_dataset_index].split(' '))
        true_action = self.dataset['label'][self.current_dataset_index]

        # Continue Reading
        if action == action_read:
            if len(self.word_chunk) < max_len: # Continue read
                reward = penalty_read
                self.current_sentence_index += 1
                self.word_chunk.append(self.dataset['text'][self.current_dataset_index].split(' ')[self.current_sentence_index])
                done = False

            else: # Force predict, reach the end
                predict = self.model(self._get_state(self.word_chunk))
                predict = self.softmax(predict.logits)
                label = torch.argmax(predict, dim=-1)

                if label == true_action:
                    reward = reward_predict_at_the_end
                else:
                    reward = penalty_predict_at_the_end
                done = True

        # Predict action
        else:
            predict = self.model(self._get_state(self.word_chunk))
            predict = self.softmax(predict.logits)
            label = torch.argmax(predict, dim=-1)

            if max_len > len(self.word_chunk): # Early predict
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

        next_state = self._get_state(self.word_chunk)
        return next_state, reward, done

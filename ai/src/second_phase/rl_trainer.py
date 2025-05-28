import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from second_phase.environment import TextEnv, split_dataset
from second_phase.reinforcement_learning import PPO, ActorCritic
import utils.model_helper as helper
import second_phase.config as configs


class RLTrainer:
    def __init__(self):
        # Load config
        self.config = configs.load_config()
        self.num_epoch = self.config['rl_training']['episode']
        self.learning_rate = self.config['rl_training']['learning_rate']
        self.num_data = self.config['rl_training']['num_data']
        self.clip_ratio = self.config['rl_training']['clip_ratio']
        self.SFT_model_path = self.config['rl_training']['SFT_model_path']
        self.k = self.config['rl_training']['k']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_tokenizer()
        self._load_dataset()
        self._init_rl_components()

    def _load_model_and_tokenizer(self):
        print('Loading SFT model...')
        self.model = BertForSequenceClassification.from_pretrained(self.SFT_model_path, num_labels=4)
        self.tokenizer = BertTokenizer.from_pretrained(self.SFT_model_path)

    def _load_dataset(self):
        print('Loading dataset...')
        dataset = load_dataset(self.config['data']['ag_news'])

        print('Tokenizing dataset...')
        train_data = split_dataset(dataset['train'], samples_per_class=self.num_data)
        train_token = self.tokenizer(train_data['text'], padding=True, return_tensors=None)
        self.train_tokens = [torch.tensor(token).to(self.device) for token in train_token['input_ids']]
        self.train_labels = torch.tensor(train_data['label']).to(self.device)

        # Clear memory
        del dataset
        del train_data

    def _init_rl_components(self):
        self.actor_critic = ActorCritic(self.model).to(self.device)
        self.optimizer = AdamW(self.actor_critic.parameters(), lr=self.learning_rate)
        self.ppo = PPO(self.actor_critic, self.optimizer, self.clip_ratio)
        self.env = TextEnv(self.train_tokens, self.train_labels, self.model, self.k, self.device)

    def train(self):
        print('Start training...')
        batch_size = self.num_data * 4
        minibatch_size = 256
        process = tqdm(range(self.num_epoch))

        for epoch in process:
            states, actions, rewards, old_log_probs = [], [], [], []
            state = self.env.reset()
            total_reward = 0.0
            count = 0

            # ===== ROLLOUT PHASE =====
            for episode in range(batch_size):
                states_batch, actions_batch, rewards_batch, old_log_probs_batch = [], [], [], []
                done = False
                eps_reward = 0
                step = 0
                # print("Episode reward: ", end='')

                while not done:
                    state = state.to(self.device)
                    with torch.no_grad():
                        action_probs, _ = self.actor_critic(state)
                    action = torch.argmax(action_probs, dim=1)

                    log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)).squeeze(1))
                    next_state, reward, done = self.env.step(action, self.tokenizer.pad_token_id)
                    # print(reward, end=f' [{step}]-> ')
                    eps_reward += reward

                    states_batch.append(state)
                    actions_batch.append(action)
                    rewards_batch.append(reward)
                    old_log_probs_batch.append(log_prob)

                    state = next_state
                    step += 1

                # print(f"{eps_reward:.2f}")
                count += 1
                total_reward += eps_reward
                process.set_description(f"Epoch {epoch+1}/{self.num_epoch}, [Rollout] Episode {episode+1}/{batch_size}, Episode Avg Reward: {total_reward/count:.2f}")
                # Pad States batch
                if count % minibatch_size == 0 or count == batch_size:
                    padded_states_batch = pad_sequence([s.squeeze(0) for s in states_batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
                    # print(padded_states_batch.shape)
                    states.append(padded_states_batch)
                    actions.append(torch.tensor(actions_batch))
                    rewards.append(rewards_batch)
                    old_log_probs.append(torch.tensor(old_log_probs_batch))
                # Next episode
                state = self.env.next_sentence()
                if state is None: break

            states = torch.cat(states).to(self.device)
            actions = torch.cat(actions).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            old_log_probs = torch.cat(old_log_probs).to(self.device)

            # ===== UPDATE POLICY PHASE =====
            for start in range(0, len(states), minibatch_size):
                end = start + minibatch_size
                self.ppo.update(
                    states[start:end],
                    actions[start:end],
                    rewards[start:end],
                    old_log_probs[start:end]
                )

            # Save model at each epoch (optional: can use early stopping/best reward logic here)
            self._save_model(epoch)

    def _save_model(self, epoch):
        print('Saving policy model...')
        helper.save_checkpoint(
            model_dir='actor_critic',
            epoch=epoch,
            model=self.actor_critic,
            optimizer=self.optimizer,
            is_the_best=True
        )
        if self.config['project']['use_gdrive']: helper.save_best_checkpoint_to_gdrive(model_dir='actor_critic')

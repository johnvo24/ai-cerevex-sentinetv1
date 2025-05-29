import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

from second_phase.environment import TextEnv, split_dataset
from second_phase.reinforcement_learning import PPO, ActorCritic
from utils.gdrive import GDrive
import utils.model_helper as helper
import second_phase.config as configs
import numpy as np


class RLTrainer:
    def __init__(self):
        # Load config
        self.config = configs.load_config()
        self.num_classes = self.config['input']['num_classes']
        self.num_epoch = self.config['rl_training']['num_epoch']
        self.learning_rate = self.config['rl_training']['learning_rate']
        self.batch_size = self.config['rl_training']['batch_size']
        self.mini_batch_size = self.config['rl_training']['mini_batch_size']
        self.max_chunk_length = self.config['rl_training']['max_chunk_length']
        self.clip_ratio = self.config['rl_training']['clip_ratio']
        self.discount_y = self.config['rl_training']['discount_y']
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
        train_data = split_dataset(dataset['train'], samples_per_class=int(self.batch_size/self.num_classes))
        train_token = self.tokenizer(train_data['text'], padding=True, return_tensors=None)
        self.train_tokens = [torch.tensor(token).to(self.device) for token in train_token['input_ids']]
        self.train_labels = torch.tensor(train_data['label']).to(self.device)
        eval_data = split_dataset(dataset['test'], samples_per_class=int(self.batch_size*0.25/self.num_classes))
        eval_token = self.tokenizer(eval_data['text'], padding=True, return_tensors=None)
        self.eval_tokens = [torch.tensor(token).to(self.device) for token in eval_token['input_ids']]
        self.eval_labels = torch.tensor(eval_data['label']).to(self.device)

        # Clear memory
        del dataset
        del train_data
        del eval_data

    def _init_rl_components(self):
        self.actor_critic = ActorCritic(self.model).to(self.device)
        self.optimizer = AdamW(self.actor_critic.parameters(), lr=self.learning_rate)
        self.ppo = PPO(self.actor_critic, self.optimizer, self.clip_ratio)
        self.env = TextEnv(self.train_tokens, self.train_labels, self.model, self.k, self.device)

    def _pad_fixed_length(self, tensors, max_length, pad_token_id):
        padded = []
        for tensor in tensors:
            tensor = tensor.view(-1)
            length = tensor.size(0)
            if length > max_length: 
                padded_tensor = tensor[:max_length]
            else:
                pad_size = max_length - length
                padding = torch.full((pad_size,), pad_token_id, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor, padding], dim=0)
            padded.append(padded_tensor)
        return torch.stack(padded, dim=0)
    
    def _comput_discounted_rewards(self, eps_rewards_list, discount_y):
        discounted_rewards = []
        reward = 0
        for r in reversed(eps_rewards_list):
            reward = r + discount_y * reward
            discounted_rewards.insert(0, reward)  # prepend
        return discounted_rewards
        

    def train(self, from_gdrive=False):
        print(f"{'='*10} TRAINING LOOP {'='*10}")
        start_epoch = -1
        if from_gdrive:
            best_checkpoint = GDrive().load_model_from_drive(file_name='best_checkpoint.tar', model_dir='actor_critic')
            self.actor_critic.load_state_dict(state_dict=best_checkpoint['model_state_dict'], strict=False)
            self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
            start_epoch = best_checkpoint['epoch']
            print(f"Loaded best_checkpoint.tar from GDrive at epoch {start_epoch}")
        self.actor_critic.train() # Set train mode

        train_avg_reward = float('-inf')
        best_eval_avg_reward = float('-inf')
        process = tqdm(range(start_epoch+1, self.num_epoch))
        for epoch in process:
            states, actions, rewards, old_log_probs = [], [], [], []
            states_batch_input_ids, states_batch_attention_mask, actions_batch, rewards_batch, old_log_probs_batch = [], [], [], [], []
            state = self.env.reset()
            total_reward = 0.0
            count = 0

            # ===== ROLLOUT PHASE =====
            for episode in range(self.batch_size):
                done = False
                eps_reward = 0
                step = 0
                # print("Episode reward: ", end='')
                eps_rewards_list = []

                while not done:
                    state.to(self.device)
                    state_input_ids = self._pad_fixed_length([state], self.max_chunk_length, self.tokenizer.pad_token_id).to(self.device)
                    state_attention_mask = (state_input_ids != 0).long().to(self.device)
                    state_inputs = {
                        'input_ids': state_input_ids,
                        'attention_mask': state_attention_mask
                    }
                    with torch.no_grad():
                        action_probs, _, _ = self.actor_critic(state_inputs)
                        dist = torch.distributions.Categorical(action_probs)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    
                    next_state, reward, done = self.env.step(action, self.tokenizer.pad_token_id)
                    # print(reward, end=f' [{step}]-> ')
                    eps_rewards_list.append(reward)
                    eps_reward += reward
                    # Append to minibatch buffer
                    states_batch_input_ids.append(state_input_ids)
                    states_batch_attention_mask.append(state_attention_mask)
                    actions_batch.append(action)
                    old_log_probs_batch.append(log_prob)
                    # Next state
                    state = next_state
                    step += 1

                eps_discounted_rewards = self._comput_discounted_rewards(eps_rewards_list, self.discount_y)
                rewards_batch.extend(eps_discounted_rewards)
                # print(f"{eps_reward:.2f}")
                count += 1
                total_reward += eps_reward
                process.set_description(f"Epoch {epoch+1}/{self.num_epoch}, [Rollout] Episode {episode+1}/{self.batch_size}, Episode Avg Reward: {total_reward/count:.2f}")
                # Pad States batch
                if count % self.mini_batch_size == 0 or count == self.batch_size:
                    # Append to rollout buffer
                    states.append({
                        'input_ids': torch.cat(states_batch_input_ids, dim=0),
                        'attention_mask': torch.cat(states_batch_attention_mask, dim=0)
                    })
                    actions.append(torch.tensor(actions_batch))
                    rewards.append(rewards_batch)
                    old_log_probs.append(torch.tensor(old_log_probs_batch))
                    # Reset batch buffer
                    states_batch_input_ids, states_batch_attention_mask, actions_batch, rewards_batch, old_log_probs_batch = [], [], [], [], []
                # Next episode
                state = self.env.next_sentence()
                if state is None: break

            # ===== UPDATE POLICY PHASE =====
            print("Avg policy loss: ", end='')
            total_policy_loss = 0
            num_minibatches = len(states)
            while states:
                state = states.pop(0)
                action = actions.pop(0).to(self.device)
                reward = torch.tensor(rewards.pop(0), dtype=torch.float32, device=self.device)
                old_log_prob = old_log_probs.pop(0).to(self.device)

                policy_loss = self.ppo.update(
                    states=state,
                    actions=action,
                    rewards=reward,
                    old_log_probs=old_log_prob
                )
                policy_loss = policy_loss.item()
                print(policy_loss, end=' -> ')
                total_policy_loss += policy_loss
            print(total_policy_loss/num_minibatches)
            print(0.2*np.exp(-0.53*(epoch)))
            
            if total_reward/count >= train_avg_reward - 0.05:
                eval_avg_reward = self._evaluate_on_eval_set()
                if eval_avg_reward > best_eval_avg_reward - 0.2*np.exp(-0.53*(epoch)): # De tai epoch 0 (-0.2) con epoch 10(-0.001)
                    if eval_avg_reward >= best_eval_avg_reward: best_eval_avg_reward = eval_avg_reward
                    print('> Saving policy model...')
                    helper.save_checkpoint(
                        model_dir='actor_critic',
                        epoch=epoch,
                        model=self.actor_critic,
                        optimizer=self.optimizer,
                        is_the_best=True
                    )
                    if self.config['project']['use_gdrive']: helper.save_best_checkpoint_to_gdrive(model_dir='actor_critic')
                else:
                    print('Model is not good to save. (eval_avg_reward)')
            else:
                print('Model is not good to save. (train_avg_reward)')
            train_avg_reward = total_reward/count


    def _evaluate_on_eval_set(self):
        print("Evaluating on eval set...")
        eval_env = TextEnv(self.eval_tokens, self.eval_labels, self.model, self.k, self.device)
        self.actor_critic.eval() # Set eval mode
        total_eval_reward = 0.0
        count = 0

        for idx, state in enumerate(self.eval_tokens):
            done = False
            state = state.to(self.device)
            eps_rewards = []
            step = 0

            while not done:
                state.to(self.device)
                state_input_ids = self._pad_fixed_length([state], self.max_chunk_length, self.tokenizer.pad_token_id).to(self.device)
                state_attention_mask = (state_input_ids != 0).long().to(self.device)
                state_inputs = {
                    'input_ids': state_input_ids,
                    'attention_mask': state_attention_mask
                }
                with torch.no_grad():
                    action_probs, _, _ = self.actor_critic(state_inputs)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.probs.argmax(dim=-1)
                
                next_state, reward, done = eval_env.step(action, self.tokenizer.pad_token_id)
                eps_rewards.append(reward)
                state = next_state
                step += 1

            count += 1
            total_eval_reward += sum(eps_rewards)
            state = eval_env.next_sentence()
            if state is None: break
            
        self.actor_critic.train()  # Trở về chế độ training
        del eval_env
        avg_eval_reward = total_eval_reward / count
        print(f"Eval Avg Reward: {avg_eval_reward:.4f}")
        return avg_eval_reward
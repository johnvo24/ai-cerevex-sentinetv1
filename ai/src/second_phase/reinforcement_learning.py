import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, model=None):
        super(ActorCritic, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.actor = nn.Linear(768, 2)
        self.critic = nn.Linear(768, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        with torch.no_grad():
            output = self.model(**x, output_hidden_states=True)
        hidden_state = output.hidden_states[-1]
        cls_token = hidden_state[:, 0, :]

        action_probs = self.softmax(self.actor(cls_token))
        state_values = self.critic(cls_token)
        pred_label = torch.argmax(output.logits, dim=1)

        return action_probs, state_values, pred_label

    def state_dict(self, *args, **kwargs):
        # Lấy toàn bộ state_dict bình thường
        full_dict = super().state_dict(*args, **kwargs)
        # Lọc bỏ key liên quan đến self.model
        filtered_dict = {k: v for k, v in full_dict.items() if not k.startswith('model.')}
        return filtered_dict


class PPO:
    def __init__(self, actorcritic, optimizer, clip_epsilon=0.2):
        self.actorcritic = actorcritic
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon

    # Incomplete loss function
    def compute_loss(self, states, actions, rewards, old_log_probs):
        # Forward pass
        action_probs, state_values, _ = self.actorcritic(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
            
        adv = rewards - state_values.detach()
        ratio = torch.exp(new_log_probs - old_log_probs)

        obj = adv*ratio
        obj_clipped = adv*torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        policy_loss = -torch.min(obj, obj_clipped).mean()

        return policy_loss

    def update(self, states, actions, rewards, old_log_probs):
        self.optimizer.zero_grad()
        loss = self.compute_loss(states, actions, rewards, old_log_probs)
        # with torch.autograd.detect_anomaly():
        loss.backward()
        self.optimizer.step()
        return loss

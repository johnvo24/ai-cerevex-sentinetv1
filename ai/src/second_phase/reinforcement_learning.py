import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, model=None):
        super(ActorCritic, self).__init__()
        self.model = model
        self.actor = nn.Linear(768, 2)
        self.critic = nn.Linear(768, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        hidden_state = output.hidden_states[-1]
        cls_token = hidden_state[:, 0, :]

        action_probs = self.softmax(self.actor(cls_token))
        state_value = self.critic(cls_token)

        return action_probs, state_value

class PPO:
    def __init__(self, actorcritic, optimizer, clip_param=0.2):
        self.actorcritic = actorcritic
        self.optimizer = optimizer
        self.clip_param = clip_param

    def compute_loss(self, states, actions, rewards, next_states):
        action_probs, state_values = self.actorcritic(states)
        old_action_probs = action_probs.detach()

        adv = rewards - state_values.detach()
        ratio = action_probs.gather(1, actions) / old_action_probs.gather(1, actions)

        obj = adv*ratio
        obj_clipped = adv*torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
        loss = -torch.min(obj, obj_clipped).mean()

        return loss

    def update(self, states, actions, rewards, next_states):
        self.optimizer.zero_grad()
        loss = self.compute_loss(states, actions, rewards, next_states)
        loss.backward()
        self.optimizer.step()

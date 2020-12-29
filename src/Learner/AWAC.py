import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.utils.train_utils import soft_update


class AWAC(nn.Module):

    def __init__(self,
                 critic: nn.Module,  # Q(s,a)
                 critic_target: nn.Module,
                 actor: nn.Module,  # pi(a|s)
                 lam: float = 0.3,  # Lagrangian parameter
                 tau: float = 5 * 1e-3,
                 gamma: float = 0.9,
                 num_action_samples: int = 1,
                 critic_lr: float = 3 * 1e-4,
                 actor_lr: float = 3 * 1e-4,
                 use_adv: bool = False):
        super(AWAC, self).__init__()

        self.critic = critic
        self.critic_target = critic_target
        self.critic_target.load_state_dict(critic.state_dict())
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(), lr=critic_lr)

        self.actor = actor
        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(), lr=actor_lr)

        assert lam > 0, "Lagrangian parameter 'lam' requires to be strictly larger than 0.0"
        self.lam = lam
        self.tau = tau
        self.gamma = gamma
        self.num_action_samples = num_action_samples
        self.use_adv = use_adv

    def get_action(self, state, num_samples: int = 1):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        return dist.sample(sample_shape=[num_samples]).T

    def update_critic(self, state, action, reward, next_states, dones):
        with torch.no_grad():
            qs = self.critic_target(next_states)  # [minibatch size x #.actions]
            sampled_as = self.get_action(next_states, self.num_action_samples)  # [ minibatch size x #. action samples]
            mean_qsa = qs.gather(1, sampled_as).mean(dim=-1, keepdims=True)  # [minibatch size x 1]
            q_target = reward + self.gamma * mean_qsa * (1 - dones)

        q_val = self.critic(state).gather(1, action)
        loss = F.mse_loss(q_val, q_target)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        # target network update
        soft_update(self.critic, self.critic_target, self.tau)

        return loss

    def update_actor(self, state, action):
        logits = self.actor(state)
        log_prob = Categorical(logits=logits).log_prob(action.squeeze()).view(-1, 1)

        with torch.no_grad():
            if self.use_adv:
                qs = self.critic_target(state)  # [#. samples x # actions]
                action_probs = F.softmax(logits, dim=-1)
                vs = (qs * action_probs).sum(dim=-1, keepdims=True)
                qas = qs.gather(1, action)
                adv = qas - vs
            else:
                adv = self.critic_target(state).gather(1, action)

            weight_term = torch.exp(1.0 / self.lam * adv)

        loss = (log_prob * weight_term).mean() * -1

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return loss

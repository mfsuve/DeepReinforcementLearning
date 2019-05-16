import torch
import numpy as np
from collections import namedtuple

from blg604ehw2.utils import process_state
from blg604ehw2.atari_wrapper import LazyFrames

Hidden = namedtuple("Hidden", "actor critic")


class BaseA3c(torch.nn.Module):
    """ Base class for Asynchronous Advantage Actor-Critic agent.
    This is a base class for both discrete and continuous
    a3c implementations.

    Arguments:
        - network: Neural network with both value and
        distribution heads
    """
    def __init__(self, network):
        super().__init__()
        self.network = network
        ### YOUR CODE HERE ###
        self.hidden_size = network.hidden_size
        ###       END      ###

    def greedy_policy(self):
        """ Return best action at the given state """
        raise NotImplementedError

    def soft_policy(self):
        """ Return a sample from the distribution of
        the given state
        """
        raise NotImplementedError

    def loss(self, transitions, last_state, is_terminal, gamma, beta):
        """ Perform gradient calculations via backward
        operation for actor and critic loses.

        Arguments:
            - transitions: List of past n-step transitions
            that includes value, entropy, log probability and
            reward for each transition
            - last_state: Next state after the given transitions
            - is_terminal: True if the last_state is a terminal state
            - gamma: Discount rate
            - beta: Entropy regularization constant
        """

        ### YOUR CODE HERE ###

        # Transtions can be either
        #   - reward, value, entropy, log probability
        #   of the states and actions
        #   - state, action
        #   from the bootstrap buffer
        #   First one is suggested!

        values, rewards, log_probs, entropies = transitions

        actor_loss, critic_loss = 0, 0
        R = torch.zeros(1, 1).to(self.device)
        if not is_terminal:
            value = self.soft_policy(last_state)[1]
            R = value.detach()

        values.append(R)

        gae = torch.zeros(1, 1).to(self.device)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            gae = gae * gamma + delta_t

            actor_loss = actor_loss - log_probs[i].sum() * gae.detach() - beta * entropies[i].sum()
############################## ilk hali #########################################################
        # R = 0#torch.zeros(1, 1).to(self.device)
        # if not is_terminal:
        #     _, val, _ = self.soft_policy(last_state)
        #     R = float(val.data)
        #
        # actor_loss = 0
        # critic_loss = 0
        #
        # for (reward, value, entropy, log_prob) in reversed(transitions):
        #     R = gamma * R + reward
        #     advantage = R - value
        #
        #     critic_loss += advantage ** 2
        #     actor_loss -= log_prob * advantage + beta * entropy

        ###       END      ###
        return actor_loss, critic_loss

    def synchronize(self, state_dict):
        """ Synchronize the agent with the given state_dict """
        self.load_state_dict(state_dict)

    def global_update(self, opt, global_agent):
        """ Update the global agent with the agent's gradients
        In order to use this method, backwards need to called beforehand
        """
        if next(self.parameters()).is_shared():
            raise RuntimeError(
                "Global network(shared) called global update!")
        for global_p, self_p in zip(global_agent.parameters(), self.parameters()):
            if global_p.grad is not None:
                continue
            else:
                global_p._grad = self_p.grad
        opt.step()

    def zero_grad(self):
        """ Clean the gradient buffers """
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @property
    def device(self):
        """ Return device name """
        return self._device

    @device.setter
    def device(self, value):
        """ Set device name and the model's
         device.
        """
        super().to(value)
        self._device = value


class ContinuousA3c(BaseA3c):
    """ Continuous action space A3C """
    def __init__(self, network):
        super().__init__(network)

    def greedy_policy(self, dist, clip_low=-1, clip_high=1):
        """ Return best action at the given state """
        ### YOUR CODE HERE ###
        # Here, I assumed "dist" to be a state and
        # hidden states for policy and value networks
        value, mu, _, (hx, cx) = self.network(dist)
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu.cpu().numpy()[0]
        return action, (hx, cx)
        ###       END      ###

    def soft_policy(self, action, clip_low=-1, clip_high=1):
        """ Sample an action  """
        ### YOUR CODE HERE ###
        # Here, I assumed "action" to be a state and
        # hidden states for policy and value networks
        value, mu, sigma, (hx, cx) = self.network(action)
        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = torch.nn.functional.softplus(sigma) + 1e-5
        eps = torch.randn(mu.size()).to(self.device)
        pi = np.array([np.pi])
        pi = torch.from_numpy(pi).float().to(self.device)

        action = (mu + sigma.sqrt() * eps).detach()

        a = (-1 * (action - mu).pow(2) / (2 * sigma)).exp()
        b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
        prob = a * b

        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        log_prob = (prob + 1e-6).log()
        return action, value, log_prob, entropy, (hx, cx)
        ###       END      ###


class DiscreteA3c(BaseA3c):
    """ Discrete action space A3C """
    def __init__(self, network):
        super().__init__(network)

    def greedy_policy(self, dist):
        """ Return best action at the given state """
        ### YOUR CODE HERE ###
        _, logit, (hx, cx) = self.network(dist)

        prob = torch.nn.functional.softmax(logit, dim=1)
        action = prob.max(1)[1].detach().cpu().numpy()[0]

        return action, (hx, cx)
        ###       END      ###

    def soft_policy(self, action):
        """ Sample an action  """
        ### YOUR CODE HERE ###
        value, logit, (hx, cx) = self.network(action)

        prob = torch.nn.functional.softmax(logit, dim=1)
        log_prob = torch.nn.functional.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)

        action = prob.multinomial(1).detach()
        log_prob = log_prob.gather(1, action)

        return action[0], value, log_prob, entropy, (hx, cx)
        ###       END      ###

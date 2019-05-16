"""
Deep Q network implementations.

Vanilla DQN and DQN with Duelling architecture,
Prioritized ReplayBuffer and Double Q learning.
"""

import torch
import numpy as np
import random
from copy import deepcopy
from collections import namedtuple

from blg604ehw2.dqn.replaybuffer import UniformBuffer
from blg604ehw2.dqn.replaybuffer import PrioirtyBuffer
from blg604ehw2.dqn.replaybuffer import Transition
from blg604ehw2.atari_wrapper import LazyFrames
from blg604ehw2.utils import process_state
from blg604ehw2.utils import normalize


class BaseDqn:
    """
    Base class for DQN implementations.

    Both greedy and e_greedy policies are defined.
    Greedy policy is a wrapper for the _greedy_policy
    method.

    Arguments:
        - nact: Number of the possible actions
        int the action space
        - buffer_capacity: Maximum capacity of the
        replay buffer
    """

    def __init__(self, nact, buffer_capacity):
        super().__init__()
        self.nact = nact
        self.buffer_capacity = buffer_capacity
        self._device = "cpu"

    def greedy_policy(self, state):
        """ Wrapper for the _greedy_policy of the
        inherited class. Performs normalization if
        the state is a LazyFrame(stack of gray images)
        and cast the state to torch tensor with
        additional dimension to make it compatible
        with the neural network.
        """
        ### Optional, You many not use this ###
        if isinstance(state, LazyFrames):
            state = np.array(state, dtype="float32")
            state = state.transpose(2, 0, 1)
            state = normalize(state)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if state.shape[0] != 1:
            state.unsqueeze_(0)
        with torch.no_grad():
            return self._greedy_policy(state)

    def e_greedy_policy(self, state, epsilon):
        """ Return action from greedy policy
        with the 1-epsilon probability and
        random action with the epsilon probability.
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.nact - 1)
        else:
            return self.greedy_policy(state)

    def push_transition(self, transition):
        """ Push transition to the replay buffer """
        raise NotImplementedError

    def update(self, batch_size):
        """ Update the model """
        raise NotImplementedError

    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        raise NotImplementedError

    @property
    def buffer_size(self):
        """ Return buffer size """
        return self.buffer.size

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


class DQN(BaseDqn, torch.nn.Module):
    """ Vanilla DQN with target network and uniform
    replay buffer. Implemantation of DeepMind's Nature
    paper.

    Arguments:
        - valuenet: Neural network to represent value
        function.
        - nact: Number of the possible actions
        int the action space
        - lr: Learning rate of the optimization
        (default=0.001)
        - buffer_capacity: Maximum capacity of the
        replay buffer (default=10000)
        - target_update_period: Number of steps for
        the target network update. After each update
        counter set to zero again (default=100)

    """

    def __init__(self, valuenet, nact, lr=0.001, buffer_capacity=10000,
                 target_update_period=100):
        super().__init__(nact, buffer_capacity)
        self.valuenet = valuenet
        self.target_net = deepcopy(valuenet)
        self.target_update_period = target_update_period
        self.target_update_counter = 0
        self.buffer = UniformBuffer(capacity=buffer_capacity)

        self.opt = torch.optim.Adam(self.valuenet.parameters(), lr=lr)

    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        ### YOUR CODE HERE ###
        # You may skip this and override greedy_policy directlyDQN

        return self.valuenet(state).argmax().item()

        ###      END       ###

    def push_transition(self, transition, *args):
        """ Push transition to the replay buffer
            Arguments:
                - transition: Named tuple of (state,
                action, reward, next_state, terminal)
        """

        self.buffer.push(transition)

    def update(self, batch_size, gamma):
        """ Update the valuenet and targetnet(if period)
        and return mean absolute td error. Process samples
        sampled from the replay buffer for q learning update.
        Raise assertion if thereplay buffer is not big
        enough for the batchsize.
        """
        assert batch_size < self.buffer.size, "Buffer is not large enough!"
        ### YOUR CODE HERE ###

        batch = self.buffer.sample(batch_size)
        batch = Transition(*map(lambda e: e.to(self.device), batch))

        state_action_values = self.valuenet(batch.state).gather(1, batch.action.long().unsqueeze(-1))

        y = torch.zeros(batch_size, device=self.device)

        non_terminal_idx = batch.terminal == 0
        non_terminal_next_states = batch.next_state.index_select(0, non_terminal_idx.nonzero().squeeze())
        y[non_terminal_idx] = self.target_net(non_terminal_next_states).max(1)[0].detach()

        expected_state_action_values = (y * gamma) + batch.reward
        expected_state_action_values = expected_state_action_values.unsqueeze(-1)

        MAE = torch.nn.L1Loss()
        MSE = torch.nn.MSELoss()

        with torch.no_grad():
            td_error = MAE(state_action_values, expected_state_action_values).item()

        loss = MSE(state_action_values, expected_state_action_values)
        self.opt.zero_grad()
        loss.backward()
        for param in self.valuenet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_period == 0:
            self.target_net.load_state_dict(self.valuenet.state_dict())
            self.target_update_counter = 0

        ###       END      ###
        return td_error  # mean absolute td error


class DuelingDoublePrioritizedDQN(BaseDqn, torch.nn.Module):
    """ DQN implementaiton with Duelling architecture,
    Prioritized Replay Buffer and Double Q learning. Double
    Q learning idea is implemented with a target network that
    is replaced with the main network at every Nth step.

    Arguments:
        - valuenet: Neural network to represent value
        function.
        - nact: Number of the possible actions
        int the action space
        - lr: Learning rate of the optimization
        (default=0.001)
        - buffer_capacity: Maximum capacity of the
        replay buffer (default=10000)
        - target_replace_period: Number of steps to
        replace value network with the target network
        (default=50)

    """

    def __init__(self, valuenet, nact, lr=0.001, buffer_capacity=10000,
                 target_replace_period=50):
        super().__init__(nact, buffer_capacity)
        ### YOUR CODE HERE ###

        self.valuenet = valuenet
        self.target_net = deepcopy(valuenet)
        self.target_update_period = target_replace_period
        self.target_update_counter = 0
        self.buffer = PrioirtyBuffer(capacity=buffer_capacity)

        self.opt = torch.optim.Adam(self.valuenet.parameters(), lr=lr)

        ###       END      ###

    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        ### YOUR CODE HERE ###

        return self.valuenet(state).argmax().item()

        ###       END      ###

    def td_error(self, trans, gamma):
        """ Return the td error, predicted values and
        target values.
        """
        # Optional but convenient
        ### YOUR CODE HERE ###
        s, a, r, ns, t = trans

        if isinstance(s, LazyFrames):
            s = normalize(np.array(s, dtype='float32').transpose(2, 0, 1))
            ns = normalize(np.array(ns, dtype='float32').transpose(2, 0, 1))

        s = torch.from_numpy(s).to(self.device).unsqueeze(0)
        a = torch.from_numpy(np.array(a, dtype='float32')).to(self.device).unsqueeze(0)
        r = torch.from_numpy(np.array(r, dtype='float32')).to(self.device).unsqueeze(0)
        ns = torch.from_numpy(ns).to(self.device).unsqueeze(0)
        t = torch.from_numpy(np.array(t, dtype='float32')).to(self.device).unsqueeze(0)

        with torch.no_grad():
            values = self.valuenet(s).gather(1, a.long().unsqueeze(-1)).squeeze()

            argmax = self.valuenet(ns).argmax(1, keepdim=True)
            target = self.target_net(ns).gather(1, argmax).squeeze()
            expected = r + (gamma * target * (1 - t))

            td_error = (values - expected.squeeze()).abs()

            del values
            del expected
            del target
            del argmax

        return td_error.detach().mean().item()

        ###       END      ###

    def push_transition(self, transition, gamma):
        """ Push transitoins and corresponding td error
        into the prioritized replay buffer.
        """
        ### YOUR CODE HERE ###

        # Remember Prioritized Replay Buffer requires
        # td error to push a transition. You need
        # to calculate it for the given trainsition

        self.buffer.push(transition, self.td_error(transition, gamma))

        ###       END      ###

    def update(self, batch_size, gamma):
        """ Update the valuenet and replace it with the
        targetnet(if period). After the td error is
        calculated for all the batch, priority values
        of the transitions sampled from the buffer
        are updated as well. Return mean absolute td error.
        """
        assert batch_size < self.buffer.size, "Buffer is not large enough!"

        ### YOUR CODE HERE ###

        # This time it is double q learning.
        # Remember the idea behind double q learning.

        idxs, batch, ws = self.buffer.sample(batch_size)
        state, action, reward, next_state, done = batch

        if state.ndim > 2:
            state = state.transpose(0, 3, 1, 2)
            state = normalize(state)
            next_state = next_state.transpose(0, 3, 1, 2)
            next_state = normalize(next_state)

        state = torch.from_numpy(state).to(self.device)
        action = torch.LongTensor(action).to(self.device).unsqueeze(-1)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        def td_error_sub(s, a, r, ns, t):
            values = self.valuenet(s).gather(1, a).squeeze()

            argmax = self.valuenet(ns).argmax(1, keepdim=True)
            target = self.target_net(ns).gather(1, argmax).squeeze()
            expected = r + (gamma * target * (1 - t))

            td_error = (values - expected).abs()
            return td_error.detach(), values, expected

        td_error, values, expected = td_error_sub(state, action, reward, next_state, done)
        td_error = td_error.mean().item()

        # Imprtance Sampling Weights
        loss = ((values - expected).pow(2) * torch.FloatTensor(ws).to(self.device)).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_period == 0:
            self.target_net.load_state_dict(self.valuenet.state_dict())
            self.target_update_counter = 0

        new_vals = td_error_sub(state, action, reward, next_state, done)[0]
        self.buffer.update_priority(idxs, new_vals.detach().cpu().squeeze().numpy())

        ###       END      ###
        return td_error  # mean absolute td error
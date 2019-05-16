"""Replay buffer implemantaions for DQN"""
from builtins import property
from collections import namedtuple
from random import sample as randsample
import numpy as np
import torch

from blg604ehw2.dqn.sumtree import SumTree

Transition = namedtuple("Transition", ("state",
                                       "action",
                                       "reward",
                                       "next_state",
                                       "terminal")
                        )


class BaseBuffer():
    """ Base class for the buffers. Push and sample
    methods need to be override. Initially start with
    an empty list(queue).

    Arguments:
        - capacity: Maximum size of the buffer
    """

    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity

    @property
    def size(self):
        """Return the current size of the buffer"""
        return len(self.queue)

    def __len__(self):
        """Return the capacity of the buffer"""
        return self.capacity

    def push(self, transition, *args, **kwargs):
        """Push transition into the buffer"""
        raise NotImplementedError

    def sample(self, batchsize, *args, **kwargs):
        """Sample transition from the buffer"""
        raise NotImplementedError


class UniformBuffer(BaseBuffer):
    """ Vanilla buffer which was used in the
    nature paper. Uniformly samples transition.

    Arguments:
        - capacity: Maximum size of the buffer
    """

    def __init__(self, capacity):
        super().__init__(capacity)
        ### YOUR CODE HERE ###
        self.pos = 0
        ###       END      ###

    def push(self, transition):
        """Push transition into the buffer"""
        ### YOUR CODE HERE ###
        if self.size < self.capacity:
            self.queue.append(None)
        self.queue[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity
        ###       END      ###

    def sample(self, batchsize):
        """ Return sample of transitions unfiromly
        from the buffer if buffer is large enough
        for the given batch size. Sample is a named
        tuple of transition where the elements are
        torch tensor.
        """
        ### YOUR CODE HERE ###

        transitions = randsample(self.queue, min(self.size, batchsize))
        return Transition(*map(torch.Tensor, zip(*transitions)))

        ###       END      ###


class PrioirtyBuffer(BaseBuffer):
    """ Replay buffer that sample tranisitons
    according to their prioirties. Prioirty
    value is most commonly td error.

    Arguments:
        - capacity: Maximum size of the buffer
        - min_prioirty: Values lower than the
        minimum prioirty value will be clipped
        - max_priority: Values larger than the
        maximum prioirty value will be clipped
    """

    def __init__(self, capacity, min_priority=0.1, max_priority=2):
        super().__init__(capacity)
        ### YOUR CODE HERE ###
        self.min_priority = min_priority
        self.max_priority = max_priority
        self.priorities = SumTree(capacity)
        self.pos = 0
        self.beta = 0.4
        self.beta_inc = (1.0 - self.beta) / 5000
        ###       END      ###

    def _clip_p(self, p):
        # You dont have to use this
        """ Return clipped priority """
        return min(max(p, self.min_priority), self.max_priority)

    def push(self, transition, priority):
        """ Push the transition with priority """
        ### YOUR CODE HERE ###

        if self.size < self.capacity:
            self.queue.append(None)
        self.queue[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

        max_priority = max(self.max_priority, self.priorities.leaves().max())
        self.priorities.push(max_priority)
        # self.priorities.push((priority + 1e-5) ** 0.6)
        ###       END      ###

    def sample(self, batch_size):
        """ Return namedtuple of transition that
        is sampled with probability proportional to
        the priority values. """
        ### YOUR CODE HERE ###

        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        p = []
        segment_size = self.priorities.total() / batch_size
        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)
            p.append(np.random.uniform(a, b))

        idxs = self.priorities.get(p)
        prob = self.priorities.leaves()[idxs] / self.priorities.total()
        w = (self.capacity * prob) ** (-self.beta)

        for idx in idxs:
            trans = self.queue[idx]
            states.append(trans.state)
            actions.append(trans.action)
            rewards.append(trans.reward)
            next_states.append(trans.next_state)
            terminals.append(trans.terminal)

        self.beta = min(1.0, self.beta + self.beta_inc)

        transition = Transition(np.stack(states).astype('float32'), actions, rewards,
                                np.stack(next_states).astype('float32'), terminals)

        return idxs, transition, w / w.max()

        ###       END      ###

    def update_priority(self, indexes, values):
        """ Update the priority value of the transition in
        the given index
        """
        ### YOUR CODE HERE ###

        self.priorities.update(indexes, (values + 1e-5) ** 0.6)

        ###       END      ###

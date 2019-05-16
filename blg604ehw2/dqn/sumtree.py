""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np


class SumTree():
    """ Binary heap with the property: parent node is the sum of
    two child nodes. Tree has a maximum size and whenever
    it reaches that, the oldest element will be overwritten
    (queue behaviour). All of the methods run in O(log(n)).

    Arguments
        - maxsize: Capacity of the SumTree

    """

    def __init__(self, maxsize):
        ### YOUR CODE HERE ###
        self.depth = int(np.ceil(np.log2(maxsize)) + 1)
        self.tree = np.zeros(2 ** self.depth - 1)
        self.maxsize = maxsize
        self.pos = 0
        self.total = lambda: self.tree[0]
        start = self.tree.size // 2
        end = start + self.maxsize
        self.leaves = lambda: self.tree[start:end]
        ###       END      ###

    def push(self, priority):
        """ Add an element to the tree and with the given priority.
         If the tree is full, overwrite the oldest element.

        Arguments
            - priority: Corresponding priority value
        """
        ### YOUR CODE HERE ###

        self.update(self.pos, priority)
        self.pos = (self.pos + 1) % self.maxsize

        ###       END      ###

    def get(self, priority):
        """ Return the node with the given priority value.
        Priority can be at max equal to the value of the root
        in the tree.

        Arguments
            - priority: Value whose corresponding index
                will be returned.
        """

        ### YOUR CODE HERE ###
        priority = np.array(priority, dtype=np.float64)
        idx = np.zeros_like(priority, dtype=np.uint32)

        for _ in range(self.depth - 1):
            idx = 2 * idx + 1
            cond = priority >= self.tree[idx]
            priority[cond] -= self.tree[idx][cond]
            idx[cond] += 1

        node = idx - self.tree.size // 2

        ###       END      ###
        return node

    def update(self, idx, value):
        """ Update the tree for the given idx with the
        given value. Values are updated via increasing
        the priorities of all the parents of the given
        idx by the difference between the value and
        current priority of that idx.

        Arguments
            - idx: Index of the data(not the tree).
            Corresponding index of the tree can be
            calculated via; idx + tree_size/2 - 1
            - value: Value for the node at pointed by
            the idx
        """
        ### YOUR CODE HERE ###

        end = lambda: idx == 0
        if isinstance(idx, np.ndarray):
            end = lambda: idx[0] == 0

        idx += self.tree.size // 2
        while True:
            self.tree[idx] = value
            if end():
                break
            right = ((idx + 1) // 2) * 2
            left = right - 1
            value = self.tree[left] + self.tree[right]
            idx = (idx - 1) // 2

        ###       END      ###

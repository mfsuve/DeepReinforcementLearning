""" Neural Networks for agents """

import torch


class FcNet(torch.nn.Module):
    """ Fully connected feature network """

    def __init__(self, nobs, n_neuron=128):
        super().__init__()
        self.fc_1 = torch.nn.Linear(nobs, n_neuron)
        self.fc_2 = torch.nn.Linear(n_neuron, n_neuron)
        self._init_weights()

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc_1(x))
        x = torch.nn.functional.relu(self.fc_2(x))
        return x

    # Optional
    def _init_weights(self):
        """Parmameter initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()



class Cnn(torch.nn.Module):
    def __init__(self, in_channels=4, out_feature=512):
        """ Convolutional feature network similar to the
		DeepMind's Nature paper.
		"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, out_feature)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.fc4(x.view(x.size(0), -1)))
        return x

    # Optional
    def _init_weights(self):
        """Parmameter initialization"""
        pass


class SimpleHead(torch.nn.Module):
    """ Linear output head """

    def __init__(self, nact, n_in):
        super().__init__()
        self.head = torch.nn.Linear(n_in, nact)
        self._init_weights()

    def forward(self, x):
        return self.head(x)

    # Optional
    def _init_weights(self):
        """Parmameter initialization"""
        pass


class DuellingHead(torch.nn.Module):
    """ Duelling output head
		Advantage and value functions are used for the
		final q values.

			Q(s, a) = V(s) + A(s, a) - sum_i(A(s, a_i))/n(A)

		As the number of actions is increased, duelling
		head performs better than the linear one.
	"""

    def __init__(self, nact, n_in):
        super().__init__()
        ### YOUR CODE HERE ###

        self.value = torch.nn.Linear(n_in, 1)
        self.advantage = torch.nn.Linear(n_in, nact)

        ###       END      ###

    def forward(self, x):
        ### YOUR CODE HERE ###

        A = self.advantage(x)
        V = self.value(x)

        return V + A - A.mean()

        ###       END      ###

    # Optional
    def _init_weights(self):
        """Parmameter initialization"""
        pass


class Network(torch.nn.Module):
    """ Merges feature and head networks """

    def __init__(self, feature_net, head_net):
        super().__init__()
        self.feature_net = feature_net
        self.head_net = head_net

    def forward(self, x, *args):
        x = self.feature_net(x)
        x = self.head_net(x, *args)
        return x


# --------- A3C Networks ----------

# You may discard below and implement heads without sequential
# elements if you want to.


class DiscreteDistHead(torch.nn.Module):
    """ Discrete Distribution generating sequential head """

    def __init__(self, in_feature, n_out):
        super().__init__()
        self.dist_gru = torch.nn.GRUCell(in_feature, 128)
        self.dist_head = torch.nn.Linear(128, n_out)

        self.value_gru = torch.nn.GRUCell(in_feature, 128)
        self.value_head = torch.nn.Linear(128, 1)

    def forward(self, x, h):
        h_a, h_c = h
        h_a = self.dist_gru(x, h_a)
        logits = self.dist_head(h_a)
        dist = torch.distributions.Categorical(logits=logits)

        h_c = self.value_gru(x, h_c)
        value = self.value_head(h_c)
        return dist, value, (h_a, h_c)


class ContinuousDistHead(torch.nn.Module):
    """ Continuous Distribution generating sequential head """

    def __init__(self, in_feature, n_out):
        super().__init__()
        self.dist_gru = torch.nn.GRUCell(in_feature, 128)
        self.dist_mu = torch.nn.Linear(128, n_out)
        self.dist_sigma = torch.nn.Linear(128, n_out)

        self.value_gru = torch.nn.GRUCell(in_feature, 128)
        self.value_head = torch.nn.GRUCell(128, 1)
        self._init_weights()

    def forward(self, x, h):
        h_a, h_c = h
        h_a = self.dist_gru(x, h_a)
        mu = self.dist_mu(h_a)
        sigma = torch.nn.functional.softplus(self.dist_sigma(h_a)) + 1e-5
        # dist = torch.distributions.Normal(loc=mu.squeeze(), scale=sigma.squeeze())

        h_c = self.value_gru(x, h_c)
        value = self.value_head(h_c)
        return value, mu, sigma, (h_a, h_c)

    def _init_weights(self):
        """Parmameter initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.GRUCell):
                m.bias_ih.data.fill_(0)
                m.bias_hh.data.fill_(0)


# This is the network that I created for the pong agent
class a3c_continuous(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(a3c_continuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 256)
        self.lrelu1 = torch.nn.LeakyReLU(0.1)
        self.fc2 = torch.nn.Linear(256, 256)
        self.lrelu2 = torch.nn.LeakyReLU(0.1)
        self.fc3 = torch.nn.Linear(256, 128)
        self.lrelu3 = torch.nn.LeakyReLU(0.1)
        self.fc4 = torch.nn.Linear(128, 128)
        self.lrelu4 = torch.nn.LeakyReLU(0.1)

        self.hidden_size = 128

        self.lstm = torch.nn.LSTMCell(128, 128)
        self.critic_linear = torch.nn.Linear(128, 1)
        self.actor_linear = torch.nn.Linear(128, action_size)
        self.actor_linear2 = torch.nn.Linear(128, action_size)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self._init_weights()
        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))

        x = x.view(1, -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), torch.nn.functional.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)

    def _init_weights(self):
        """Parmameter initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, 0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()


# This is the network that I created for the breakout agent
class a3c_discrete(torch.nn.Module):
    def __init__(self, stack_size, action_size):
        super(a3c_discrete, self).__init__()
        self.conv1 = torch.nn.Conv2d(stack_size, 32, 6, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=2)
        self.lstm = torch.nn.LSTMCell(64 * 4 * 4, 256)
        self.lrelu = torch.nn.LeakyReLU(0.1)

        self.hidden_size = 256

        self.critic_linear = torch.nn.Linear(self.hidden_size, 1)
        self.actor_linear = torch.nn.Linear(self.hidden_size, action_size)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self._init_weights()
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.lrelu(self.conv1(inputs))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    def _init_weights(self):
        """Parmameter initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, 0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

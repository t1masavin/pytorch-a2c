from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def init(module, weight_init, bias_init, gain=1):
    """

    :param module: module to initialize
    :param weight_init: initialization scheme
    :param bias_init: bias initialization scheme
    :param gain: gain for weight initialization
    :return: initialized module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ConvBlock(nn.Module):

    def __init__(self, ch_in=4) -> None:
        super().__init__()

        #const
        self.num_filter = 32
        self.kernel_size = 3
        self.stride = 2
        self.pad = self.kernel_size // 2

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        #layers
        self.conv1 = init_(nn.Conv2d(
            ch_in, self.num_filter, self.kernel_size, self.stride, self.pad))
        self.conv2 = init_(nn.Conv2d(
            self.num_filter, self.num_filter, self.kernel_size, self.stride, self.pad))
        self.conv3 = init_(nn.Conv2d(
            self.num_filter, self.num_filter, self.kernel_size, self.stride, self.pad))
        self.conv4 = init_(nn.Conv2d(
            self.num_filter, self.num_filter, self.kernel_size, self.stride, self.pad))

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = nn.MaxPool2d(2)(x)
        # x = nn.AdaptiveMaxPool2d(2)(x)
        # print(x.view(x.shape[0], -1).shape)
        return x.view(x.shape[0], -1)


class FeatureEncoderNet(nn.Module):
    
    def __init__(self, n_stack, input_size) -> None:
        super().__init__()

        #const
        self.input_size = input_size
        self.hidden_size = 350

        self.conv = ConvBlock(ch_in=n_stack)
        self.gru = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)

    # @torch.no_grad
    def reset_lstm(self, buffer_size=None, reset_indices=None):
        with torch.no_grad():
            if reset_indices is None:
                # set device to that of the underlying network
                # (it does not matter, the device of which layer is queried)
                self.h_t1  = torch.zeros(buffer_size, self.hidden_size, device=self.gru.weight_ih.device)
            else:
                # set device to that of the underlying network
                # (it does not matter, the device of which layer is queried)
                resetTensor = torch.as_tensor(reset_indices.astype(np.uint8), device=self.gru.weight_ih.device)
            
                if resetTensor.sum():
                    self.h_t1 = (1 - resetTensor.view(-1, 1)).float() * self.h_t1
                    # self.c_t1 = (1 - resetTensor.view(-1, 1)).float() * self.c_t1

    def forward(self, x):
        """
        In: [s_t]
            Current state (i.e. pixels) -> 1 channel image is needed

        Out: phi(s_t)
            Current state transformed into feature space

        :param x: input data representing the current state
        :return:
        """    
        x = self.conv(x)
        # x = x.view(-1, self.input_size)
        # h_t1 is the output
        self.h_t1 = self.gru(x, (self.h_t1))
        return self.h_t1
    

class A2CNet(nn.Module):
    def __init__(self, n_stack, num_actions, input_size=192, writer=None) -> None:
        """
        Implementation of the Advantage Actor-Critic (A2C) network

        :param n_stack: number of frames stacked
        :param num_actions: size of the action space, pass env.action_space.n
        :param in_size: input size of the LSTMCell of the FeatureEncoderNet
        """
        super().__init__()

        self.writer = writer

        #const
        self.input_size = input_size
        self.num_actions = num_actions

        self.feature_encoder = FeatureEncoderNet(n_stack=n_stack, input_size=input_size)
        #Q
        self.actor = nn.Linear(in_features=self.feature_encoder.hidden_size, out_features=num_actions)
        #V
        self.critic = nn.Linear(in_features=self.feature_encoder.hidden_size, out_features=1)
    
    def set_reccurent_buffers(self, buffer_size):
        """
        Initializes LSTM buffers with the proper size,
        should be called after instatiation of the network.

        :param buf_size: size of the recurrent buffer
        :return:
        """
        self.feature_encoder.reset_lstm(buffer_size=buffer_size)
    
    def reset_reccurent_buffers(self, reset_indices):
        """

        :param reset_indices: boolean numpy array containing True at the indices which
                              should be reset
        :return:
        """
        self.feature_encoder.reset_lstm(reset_indices=reset_indices)
    
    def forward(self, state):

        #encode the state
        feature = self.feature_encoder(state)
        # feature = torch.sigmoid(feature)
        #calculate policy and value function
        policy = self.actor(feature)
        # policy = torch.sigmoid(policy)
        value = self.critic(feature)

        if self.writer is not None:
            self.writer.add_histogram('feature', feature.detach())
            self.writer.add_histogram('policy', policy.detach())
            self.writer.add_histogram('value', value.detach())
        return policy, torch.squeeze(value), feature
    
    def get_action(self, state):

        policy, value, feature = self(state)
        cat = Categorical(logits=policy)
        action = cat.sample()
        return(action, cat.log_prob(action), cat.entropy().mean(), value, feature)
    
import torch
import torch.nn as nn
class RNN_QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(RNN_QNetwork, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        q_values = self.fc(out[:, -1, :])  # استفاده از آخرین خروجی RNN
        return q_values, h
import torch
import torch.nn as nn
class LSTM_QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(LSTM_QNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_actions)
    def forward(self, x, hidden):
        x = torch.relu(self.fc(x))       # استخراج ویژگی اولیه
        x, hidden = self.lstm(x, hidden) # پردازش ترتیبی
        q_values = self.out(x)           # پیش‌بینی مقدار Q
        return q_values, hidden
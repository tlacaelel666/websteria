import torch.nn as nn

class TimeInterpretator(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(TimeInterpretator, self).__init__()
    self.hidden_size = hidden_size
    self.rnn = nn.RNN(input_size, hidden_size[0], batch_first=True)
    self.fc = nn.Linear(hidden_size[0], output_size)
  def forward(self, x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out

import torch
import torch.nn as nn
from Config import *
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, Config.embedding_dim)
        self.lstm = nn.LSTM(input_size=Config.embedding_dim, hidden_size=Config.hidden_dim,
                            bidirectional=Config.bidirectional, num_layers=Config.layer_size, dropout=0.5)
        self.classify = nn.Linear(Config.hidden_dim * Config.num_direction, tag_size)

    def forward(self, input):

        # seq_len = 60  句子长度

        input = self.embed(input)  # [64, 60, 100]  [batch_size, seq_len, hidden_dim]
        input = input.permute(1, 0, 2)

        out, _ = self.lstm(input)  # [60, 64, 100]  [seq_len, batch_size, hidden_dim]

        out = out[-1, :, :]    # [64, 100]
        out = out.view(-1, Config.hidden_dim * Config.num_direction)    # [64, 100]  [batch_size, seq_len]
        out = self.classify(out)    # [64, 2]

        return out
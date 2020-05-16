import torch
import torch.nn as nn
from Config import *
from torch.autograd import Variable

class LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(LSTM_Attention, self).__init__()
        self.embed = nn.Embedding(vocab_size, Config.embedding_dim)
        self.lstm = nn.LSTM(input_size=Config.embedding_dim, hidden_size=Config.hidden_dim, bidirectional=Config.bidirectional)
        self.classify = nn.Linear(Config.hidden_dim * Config.num_direction, tag_size)

        self.w_omega = Variable(torch.zeros(Config.hidden_dim * Config.num_direction, Config.attention_size))
        self.u_omega = Variable(torch.zeros(Config.attention_size))

    def attention_net(self, out):

        # out:[seq_len, batch_size, hidden_dim * num_direction]

        output_reshape = torch.Tensor.reshape(out, [-1, Config.hidden_dim * Config.num_direction])   # [seq_len * batch_size, hidden_dim * num_direction]

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))  # [seq_len * batch_size, attention_size]

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))    # [seq_len * batch_size, 1]

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, Config.sequence_length])    # [batch_size, seq_len]

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])    # [batch_size, seq_len]

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, Config.sequence_length, 1])    # [batch_size, seq_len, 1]

        state = out.permute(1, 0, 2)    # [batch_size, seq_len, hidden_dim * num_direction]

        attn_output = torch.sum(state * alphas_reshape, 1)    # [batch_size, hidden_dim * num_direction]

        return attn_output


    def forward(self, input):

        # seq_len = 60  句子长度

        input = self.embed(input)  # [64, 60, 100]  [batch_size, seq_len, hidden_dim]
        input = input.permute(1, 0, 2)

        out, _ = self.lstm(input)  # [60, 64, 100]  [seq_len, batch_size, hidden_dim * num_direction]

        attn_output = self.attention_net(out)

        out = self.classify(attn_output)  # [64, 2]

        return out
# encoding:utf-8
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import logging

INF = 1e30

class Policy(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, activation='relu'):
        super(Policy, self).__init__()
        self.args = args
        self.use_cuda = torch.cuda.is_available() if args.use_gpu else False
        self.logger = logging.getLogger("GACM")
        self.embed_size = args.embed_size
        self.gru_hidden_size = args.gru_hidden_size
        self.critic_hidden_size = args.critic_hidden_size
        self.dropout_rate = args.dropout_rate
        self.max_d_num = args.max_d_num
        self.encode_gru_num_layer = 1
        self.query_size = query_size
        self.doc_size = doc_size
        self.vtype_size = vtype_size

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.query_embedding = nn.Embedding(query_size, self.embed_size)
        self.doc_embedding = nn.Embedding(doc_size, self.embed_size)
        self.vtype_embedding = nn.Embedding(vtype_size, self.embed_size // 2)
        self.action_embedding = nn.Embedding(2, self.embed_size // 2)


        self.actor_gru = nn.GRU(self.embed_size * 3, self.gru_hidden_size,
                                batch_first=True, num_layers=self.encode_gru_num_layer)
        self.critic_gru = nn.GRU(self.embed_size * 3, self.gru_hidden_size,
                                batch_first=True, num_layers=self.encode_gru_num_layer)

        # actor
        self.actor_linear = nn.Linear(self.gru_hidden_size, 2)
        self.softmax = nn.Softmax(dim = -1)

        # critic
        self.critic_layers = nn.ModuleList()
        last_dim = self.gru_hidden_size
        for nh in self.critic_hidden_size:
            self.critic_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.critic_linear = nn.Linear(last_dim, 1)

    def forward(self, query, doc, vtype, action, actor_rnn_state=None, critic_rnn_state=None):
        batch_size = query.size()[0]
        doc_num = doc.size()[1]
        if actor_rnn_state is None or critic_rnn_state is None:
            actor_rnn_state = Variable(torch.zeros(1, batch_size, self.gru_hidden_size))
            critic_rnn_state = Variable(torch.zeros(1, batch_size, self.gru_hidden_size))
            if self.use_cuda:
                actor_rnn_state, critic_rnn_state = actor_rnn_state.cuda(), critic_rnn_state.cuda()

        query_embed = self.query_embedding(query)  # batch_size, 11, embed_size
        doc_embed = self.doc_embedding(doc)  # batch_size, 11, embed_size
        vtype_embed = self.vtype_embedding(vtype)  # batch_size, 11, embed_size/2
        action_embed = self.action_embedding(action)  # batch_size, 11, embed_size/2
        gru_input = torch.cat((query_embed, doc_embed, vtype_embed, action_embed), dim=2)

        actor_outputs, actor_rnn_state = self.actor_gru(gru_input, actor_rnn_state)
        actor_outputs = self.dropout(actor_outputs)
        logits = self.softmax(self.actor_linear(actor_outputs)).view(batch_size, doc_num, 2)
        if logits.shape[1] > 1:
            logits = logits[:,1:,:]

        critic_outputs, critic_rnn_state = self.critic_gru(gru_input, critic_rnn_state) # share gru also works
        x = critic_outputs
        for layer in self.critic_layers:
            x = self.activation(layer(self.dropout(x)))
        value = torch.squeeze(self.critic_linear(x), dim=-1)
        if value.shape[1] > 1:
            value = value[:, 1:]
        return logits, value, actor_rnn_state, critic_rnn_state


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    class config():
        def __init__(self):
            self.embed_size = 300
            self.hidden_size = 150
            self.dropout_rate = 0.2

    args = config()
    model = Policy(args, 10, 20, 30)
    q = Variable(torch.zeros(8, 11).long())
    d = Variable(torch.zeros(8, 11).long())
    v = Variable(torch.zeros(8, 11).long())
    a = Variable(torch.zeros(8, 11).long())
    model.forward(q, d, v, a)  
    print (count_parameters(model))

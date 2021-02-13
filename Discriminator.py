# encoding:utf-8
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import logging

INF = 1e30

class Discriminator(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, activation='relu'):
        super(Discriminator, self).__init__()
        self.args = args
        self.use_cuda = torch.cuda.is_available() if args.use_gpu else False
        self.logger = logging.getLogger("GACM")
        self.embed_size = args.embed_size   # 300 as default
        self.gru_hidden_size = args.gru_hidden_size # 150 as default
        self.critic_hidden_size = args.critic_hidden_size # [256,256] as default
        self.dropout_rate = args.dropout_rate
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

        self.query_embedding = nn.Embedding(query_size, self.embed_size)
        self.doc_embedding = nn.Embedding(doc_size, self.embed_size)
        self.vtype_embedding = nn.Embedding(vtype_size, self.embed_size // 2)
        self.action_embedding = nn.Embedding(2, self.embed_size // 2)

        self.gru = nn.GRU(self.embed_size * 3, self.gru_hidden_size,
                            batch_first=True, num_layers=self.encode_gru_num_layer)

        self.output_linear = nn.Linear(self.gru_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)


    def forward(self, query, doc, vtype, action, rnn_state=None):
        batch_size = query.size()[0]
        max_doc_num = doc.size()[1]
        if rnn_state is None:
            rnn_state = Variable(torch.zeros(1, batch_size, self.gru_hidden_size))
            if self.use_cuda:
                rnn_state = rnn_state.cuda()

        query_embed = self.query_embedding(query)  # batch_size, 11, embed_size
        doc_embed = self.doc_embedding(doc)  # batch_size, 11, embed_size
        vtype_embed = self.vtype_embedding(vtype)  # batch_size, 11, embed_size /2
        action_embed = self.action_embedding(action)  # batch_size, 11, embed_size / 2
        gru_input = torch.cat((query_embed, doc_embed, vtype_embed, action_embed), dim=2)

        outputs, rnn_state = self.gru(gru_input, rnn_state)
        logits = self.sigmoid(self.output_linear(self.dropout(outputs))).view(batch_size, max_doc_num)[:, 1:]

        return logits

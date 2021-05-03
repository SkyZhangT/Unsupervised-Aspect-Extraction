import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, maxlen, emb_dim, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 b=True, **kwargs):
        super(Attention, self).__init__()
        self.steps = maxlen    # maxlen
        self.b = b

        # Initialize weight matrix and bias matrix
        self.weights = torch.nn.Parameter(nn.init.xavier_uniform_(torch.empty(emb_dim, emb_dim), gain=nn.init.calculate_gain('relu')).cuda())  # W [emb_dim, emb_dim]
        if self.b:
            self.bias = torch.nn.Parameter(nn.init.zeros_(torch.empty(1)).cuda())

    def forward(self, e, y, mask=None):
        # e [batch_size, maxlen, emb_dim]
        # y [batch_size, emb_dim]

        # Tensor computation
        y = torch.mm(self.weights, y.t())        # y [emb_dim, batch_size]
        y = y.t()                           # y [batch_size, emb_dim]
        y = y.unsqueeze(-2).repeat(1, self.steps, 1)      # y [batch_size, maxlen, emb_dim
        d = torch.sum(e*y, axis=-1)             # d [batch_size, maxlen]

        if self.bias:
            bias = self.bias.repeat(self.steps)       # b [maxlen]
            d += self.bias                         
        
        a = torch.exp(torch.tanh(d))

        if mask is not None:
            mask = mask.type(torch.FloatTensor)

        # how to add epsilon?
        a /= (torch.sum(a, 1, keepdims=True) + 1e-8).type(torch.FloatTensor).cuda()
        return a
        

class WeightedSum(nn.Module):
    def forward(self, e, att):
        att = att.unsqueeze(-1)

        weighted_input = e
        
        return torch.sum(weighted_input, 1)

class WeightedAspectEmb(nn.Module):
    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 weights=None, dropout=0.):
        super(WeightedAspectEmb, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.dropout = dropout

        # self.W_constraint = constraints.get(W_constraint)
        # self.W_regularizer = regularizers.get(W_regularizer)
        # self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights

        self.weights = torch.nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, output_dim), gain=nn.init.calculate_gain('relu')).cuda())  # W [emb_dim, emb_dim]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def forward(self, x):
        return torch.mm(x, self.weights)

class Average(nn.Module):
    def forward(self, x, mask=None):
        denom = 1
        if mask is not None:
            mask = mask.type(torch.FloatTensor)
            mask = mask.unsqueeze()
            x = x * mask
            denom = torch.sum(mask, -2)
            
        # why divide by None
        return torch.sum(x, -2) / denom
    
class MaxMargin(nn.Module):
    def forward(self, z_s, z_n, r_s, mask=None):
        z_s = z_s / (1e-8 + torch.sqrt(torch.sum(torch.square(z_s), axis=-1, keepdims=True)))       # z_s [batch_size, emb_dim]
        z_n = z_n / (1e-8 + torch.sqrt(torch.sum(torch.square(z_n), axis=-1, keepdims=True)))       # z_n [batch_size, neg_size, emb_dim]
        r_s = r_s / (1e-8 + torch.sqrt(torch.sum(torch.square(r_s), axis=-1, keepdims=True)))       # r_s [batch_size, emb_dim]
 
        steps = z_n.shape[1]    # neg size

        pos = torch.sum(z_s*r_s, axis=-1, keepdims=True).repeat(1, steps)       # pos [batch_size, neg_size]


        r_s = r_s.unsqueeze(1).repeat(1, steps, 1)      # r_s [batch_size, neg_size, emb_dim]
        neg = torch.sum(z_n*r_s, axis=-1)               # neg [batch_size, neg_size]

        zeros = torch.zeros(pos.shape, dtype=torch.float32).cuda()
        loss = torch.sum(torch.maximum(zeros, (1. - pos + neg)), axis=-1, keepdims=True)
        return loss
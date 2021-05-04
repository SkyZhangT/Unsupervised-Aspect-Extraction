import torch
import logging
import torch.nn as nn
from my_layers import Average, Attention, WeightedSum, WeightedAspectEmb, MaxMargin

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, args, maxlen, vocab):
        super(Model, self).__init__()
        self.vocab_size = len(vocab)
        self.word_emb = nn.Embedding(self.vocab_size, args.emb_dim)
        self.ortho_reg = args.ortho_reg
        
        ### do not update weight of word_emb layer
        self.word_emb.weight.requires_grad = False

        self.activation = nn.Softmax(dim=1)
        self.dense = nn.Linear(args.emb_dim, args.aspect_size)
        self.avg = Average()
        self.att = Attention(maxlen, args.emb_dim)
        self.ws = WeightedSum()
        self.wae = WeightedAspectEmb(args.aspect_size, args.emb_dim)
        self.mmLoss = MaxMargin()

        ### Word embedding and aspect embedding initialization ######
        if args.emb_path:
            from w2vEmbReader import W2VEmbReader as EmbReader
            emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
            logger.info('Initializing word embedding matrix')
            self.word_emb.weight = torch.nn.Parameter(emb_reader.get_emb_matrix_given_vocab(vocab, self.word_emb.weight), requires_grad=False)
            logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
            self.wae.weights = torch.nn.Parameter(emb_reader.get_aspect_matrix(args.aspect_size).cuda())
    
    def calc_ortho_reg(self, weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = weight_matrix / (1e-8 + torch.sqrt(torch.sum(torch.square(weight_matrix), dim=-1, keepdims=True)))
        reg = torch.sum(torch.square(torch.mm(w_n, w_n.t()) - torch.eye(w_n.shape[0]).cuda()))
        return self.ortho_reg*reg
                                                                    
    def forward(self, sentence_input, neg_input):
        if neg_input == None and self.training==True:
            raise Exception("training without neg samples")

        ##### Compute sentence representation #####
        e_w = self.word_emb(sentence_input)             # e_w [batch_size, maxlen, emb_dim]
        y_s = self.avg(e_w)                            # y_s [batch_size, emb_dim]
        att_weights = self.att(e_w, y_s)                # att_weights [batch_size, max_len]
        z_s = self.ws(e_w, att_weights)                 # z_s [batch_size, emb_dim]

        ##### Reconstruction #####
        p_t = self.dense(z_s)                           # p_t [batch_size, aspect_size]
        p_t = self.activation(p_t)                       # p_t [batch_size, aspect_size]

        if neg_input == None and self.training==False:
            return att_weights, p_t

        ##### Compute representations of negative instances #####
        e_neg = self.word_emb(neg_input)                # e_neg [batch_size, neg_size, maxlen, emb_dim]
        z_n = self.avg(e_neg)                          # z_n [batch_size, neg_size, emb_dim]
        
        r_s = self.wae(p_t)                             # r_s [batch_size, aspect_size]
        loss = self.mmLoss(z_s, z_n, r_s)               # loss [batch_size]

        reg = self.calc_ortho_reg(self.wae.weights)

        output= {"max_margin": loss, "loss": torch.mean(loss) + reg}

        return output
    
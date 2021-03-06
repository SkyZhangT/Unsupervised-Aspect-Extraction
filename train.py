import argparse
import logging
import numpy as np
from time import time
import utils as U
import codecs
import torch.nn as nn
from transformers import AdamW
import tqdm
from model import Model
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import math
import random

logging.basicConfig(
                    #filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14, help="The number of aspects specified by users (default=14)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant', help="domain of the corpus {restaurant, beer}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")

args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain
U.mkdir_p(out_dir)
U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.domain in {'restaurant', 'beer'}

if args.seed > 0:
    np.random.seed(args.seed)


# ###############################################################################################################################
# ## Prepare data
# #
import reader as dataset
from tqdm import tqdm

vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)


print(f'Number of training examples: {len(train_x)}')
print(f'Length of vocab: {len(vocab)}')
print(f'Padding length of training examples: {overall_maxlen}')

def sentence_batch_generator(data, batch_size, maxlen):
    data_length = len(data)
    indices = list(range(data_length))
    n_batch = data_length / batch_size
    batch_count = 0
    random.shuffle(indices)

    while True:
        if batch_count >= n_batch:
            random.shuffle(indices)
            batch_count = 0

        index_batch = indices[batch_count*batch_size: min((batch_count+1)*batch_size, data_length)]
        batch = []
        for i in index_batch:
            padding_length = maxlen - len(data[i])
            batch.append([0] * padding_length + data[i])
        batch_count += 1
        yield torch.LongTensor(batch)

def get_neg_batch(data, batch_size, neg_size, maxlen):
    while True:
        indices = np.random.choice(len(data), batch_size * neg_size)
        batch = []
        for i in indices:
            padding_length = maxlen - len(data[i])
            batch.append([0] * padding_length + data[i])
        return torch.LongTensor(batch)

###############################################################################################################################
## Building model
model = Model(args, overall_maxlen, vocab)
if torch.cuda.is_available():
    model.cuda()


###############################################################################################################################
## Optimizaer algorithm
#
params = [ param for param in model.parameters() if param.requires_grad == True]
optimizer = AdamW(params=params ,lr=0.001, betas=[0.9, 0.999], eps=1e-08)

###############################################################################################################################
## Training
#
sen_gen = sentence_batch_generator(train_x, args.batch_size, overall_maxlen)

batches_per_epoch = math.ceil(len(train_x)/args.batch_size)

min_loss = float('inf')
print(f"Start training the model.")
model.train()
torch.autograd.set_detect_anomaly(True)
for ii in range(args.epochs):
    t0 = time()
    loss, max_margin_loss = 0., 0.

    for i in tqdm(range(batches_per_epoch)):
        sen_batch = next(sen_gen)
        
        optimizer.zero_grad()

        neg_batch = get_neg_batch(train_x, sen_batch.shape[0], args.neg_size, overall_maxlen)
        if torch.cuda.is_available():
            sen_batch = sen_batch.cuda()
            neg_batch = neg_batch.cuda()
        neg_batch = neg_batch.reshape(sen_batch.shape[0], args.neg_size, overall_maxlen)

        output = model(sen_batch, neg_batch)
        output["loss"].backward()
        optimizer.step()

        loss += output["loss"] / batches_per_epoch

    
    print(f"Epoch {ii} finished, loss={loss.data}.")
    if loss < min_loss:
        torch.save(model.state_dict(), out_dir+'/aspect.log')

################ Evaluation ####################################
model.eval()

test_data = []
for line in test_x:
    padding_length = overall_maxlen - len(line)
    test_data.append([0] * padding_length + line)
test_data = torch.tensor(test_data)
if torch.cuda.is_available():
    test_data = test_data.cuda()

# todo


## Create a dictionary that map word index to word 
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w


def eval(data):
    model.eval()
    att_weights, prob = model(data, None)
    return att_weights.cpu(), prob.cpu()

att_weights, aspect_probs = eval(test_data)

## Save attention weights on test sentences into a file 
att_out = codecs.open(out_dir + '/att_weights', 'w', 'utf-8')
print('Saving attention weights on test sentences...')
for c in range(len(test_x)):
    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i!=0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overall_maxlen-line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' '+str(np.around(weights[j].detach().numpy(), 3)) + '\n')
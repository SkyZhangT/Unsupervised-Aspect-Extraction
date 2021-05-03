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
for index, line in tqdm(enumerate(train_x)):
    padding_length = overall_maxlen - len(line)
    train_x[index] = [0] * padding_length + line

for index, line in tqdm(enumerate(test_x)):
    padding_length = overall_maxlen - len(line)
    test_x[index] = [0] * padding_length + line

print(f'Number of training examples: {len(train_x)}')
print(f'Length of vocab: {len(vocab)}')
print(f'Padded train set length {len(train_x[0])}')
print(f'Padded test set length {len(test_x[0])}')


def convert_data_to_loader(data, batch_size):
    dataset = TensorDataset(torch.tensor(data))
    sampler = RandomSampler(dataset)
    return dataset, DataLoader(dataset, sampler=sampler, batch_size=batch_size)

###############################################################################################################################
## Building model
model = Model(args, overall_maxlen, vocab)
if torch.cuda.is_available():
    model.cuda()


###############################################################################################################################
## Optimizaer algorithm
#
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
]
optimizer = AdamW(params=model.parameters() ,lr=0.001, betas=[0.9, 0.999], eps=1e-08)

###############################################################################################################################
## Training
#

train_set, train_dataloader = convert_data_to_loader(train_x, args.batch_size)

batches_per_epoch = math.ceil(len(train_x)/args.batch_size)

min_loss = float('inf')
model.train()
torch.autograd.set_detect_anomaly(True)
for ii in range(args.epochs):
    t0 = time()
    loss, max_margin_loss = 0., 0.

    for step, sen_batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        cur_batch_size = sen_batch[0].shape[0]
        neg_batch_size = cur_batch_size * args.neg_size
        indices = np.random.choice(len(train_x), neg_batch_size)
        negset = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, indices), batch_size=neg_batch_size)
        neg = next(iter(negset))


        if torch.cuda.is_available():
            sen_batch = sen_batch[0].cuda()
            neg = neg[0].cuda()
        neg = neg.reshape(cur_batch_size, args.neg_size, overall_maxlen)

        output = model(sen_batch, neg)
        output["loss"].backward()
        optimizer.step()

        loss += output["loss"] / batches_per_epoch

    print(loss.data)
    if loss < min_loss:
        torch.save(model.state_dict(), out_dir+'/aspect.log')

################ Evaluation ####################################
model.eval()

test_data = torch.tensor(test_x)
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
    att_out.write(str(c) + '\t aspect_probs:' + str(aspect_probs[c]) + '\n')

    word_inds = [i for i in test_x[c] if i!=0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overall_maxlen-line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' '+str(np.around(weights[j].detach().numpy(), 3)) + '\n')
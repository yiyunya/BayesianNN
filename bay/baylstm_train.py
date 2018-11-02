import numpy as np
import argparse
import torch
import torch.nn as nn
from lstm_model import LSTM_Text
from torch.nn.functional import normalize  # noqa: F401

from torch.autograd import Variable

import pyro
from pyro.distributions import Normal, Bernoulli  # noqa: F401
from pyro.infer import SVI
from pyro.optim import Adam

"""
Bayesian Regression
Learning a function of the form:
    y = wx + b
"""


parser = argparse.ArgumentParser(description='LSTM text classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train [default: 32]')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training [default: 16]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', default=0, action='store_true',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='./LSTM_Text.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/corpus.pt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=64,
                    help='number of embedding dimension [default: 64]')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='number of lstm hidden dimension [default: 128]')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='biLSTM layer numbers')
parser.add_argument('--bidirectional', action='store_true',
                    help='If True, becomes a bidirectional LSTM [default: False]')

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able

from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.vocab_size = data['dict']['vocab_size']
args.label_size = data['dict']['label_size']

training_data = DataLoader(
             data['train']['src'],
             data['train']['label'],
             args.max_len,
             batch_size=args.batch_size,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['label'],
              args.max_len,
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda)
print(training_data.sents_size)


# generate toy dataset
def build_linear_dataset(N, p, noise_std=0.01):
    X = np.random.rand(N, p)
    # use random integer weights from [0, 7]
    w = np.random.randint(8, size=p)
    # set b = 1
    y = np.matmul(X, w) + np.repeat(1, N) + np.random.normal(0, noise_std, size=N)
    y = y.reshape(N, 1)
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)


# NN with one linear layer
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        # x * w + b
        return self.linear(x)


N = training_data.sents_size  # size of toy data
p = 128  # number of features
bs = args.batch_size
softplus = nn.Softplus()
regression_model = RegressionModel(p)
lstm_model = LSTM_Text(args)


def model(data):
    # Create unit normal priors over the parameters
    priors = {}
    for name, para in lstm_model.named_parameters():
        mu = Variable(torch.zeros(para.size())).type_as(para.data)
        sigma = Variable(torch.ones(para.size())).type_as(para.data)
        prior = Normal(mu, sigma)
        priors[name] = prior
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", lstm_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.iarange("map", N, subsample=data):
        x_data = data[:,:-1*args.label_size]
        y_data = data[:,-1*args.label_size:]
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data)[0]
        pyro.sample("obs",
                    Normal(prediction_mean,
                           Variable(torch.ones(bs,args.label_size)).type_as(prediction_mean)),
                    obs=y_data.type(torch.FloatTensor))


def guide(data):
    dists = {}
    for name,para in lstm_model.named_parameters():
        mu = Variable(torch.randn(para.size()).type_as(para.data), requires_grad=True)
        log_sig = Variable((-3.0 * torch.ones(para.size()) + 0.05 * torch.randn(para.size())).type_as(para.data), requires_grad=True)
        # register learnable params in the param store
        m_param = pyro.param('guide_mean_'+name, mu)
        s_param = softplus(pyro.param('guide_log_sigma_'+name, log_sig))
        # gaussian guide distributions
        dist = Normal(m_param, s_param)
        dists[name] = dist
    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", lstm_model, dists)
    # sample a regressor
    return lifted_module()


# instantiate optim and inference objects
optim = Adam({"lr": 0.001})
svi = SVI(model, guide, optim, loss="ELBO")


def label_process(label,label_size):
    l = np.zeros((len(label),label_size))
    for i in range(len(label)):
        j = label.data.tolist()[i]
        l[i][j] = 1
    l = Variable(torch.from_numpy(l).type(torch.LongTensor))
    return l



# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
accuracy = []


def evaluate():
    criterion = torch.nn.CrossEntropyLoss()
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    pred = Variable(torch.zeros(bs,args.label_size))
    for data, label in tqdm(validation_data, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        # hidden = repackage_hidden(hidden)
        sampled_reg_model = guide(None)
        pred = pred + sampled_reg_model(data)[0]
        loss = criterion(pred, label)
        eval_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    return eval_loss[0]/_size, corrects, corrects/_size * 100.0, _size

def train():

    total_loss = 0.0

    for data, label in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=False):
        # optimizer.zero_grad()
        # # hidden = repackage_hidden(hidden)
        # target, hidden = rnn(data)
        # loss = criterion(target, label)
        #
        # loss.backward()
        # optimizer.step()
        tmp_label = label_process(label, args.label_size)
        batch_data = torch.cat((data,tmp_label), 1)
        loss = svi.step(batch_data)

        total_loss += loss
    return total_loss/training_data.sents_size

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss*1000.)
        accuracy.append(acc)

        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 90)
        # if not best_acc or best_acc < corrects:
        #     best_acc = corrects
        #     model_state_dict = rnn.state_dict()
        #     model_source = {
        #         "settings": args,
        #         "model": model_state_dict,
        #         "src_dict": data['dict']['train']
        #     }
        #     torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

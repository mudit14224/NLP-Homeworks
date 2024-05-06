import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

def read_corpus(filename,tokenizer):
    seq = []
    with open(filename,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    # Changed the dot product to euclidean distance metric
    # scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    q_ex = q.unsqueeze(2)
    k_ex = k.unsqueeze(2)

    dist = torch.sum((q_ex - k_ex)**2, dim = -1)
    # -ve because lower distances mean higher similarity
    scores = -dist
    
    if mask is not None:
        # mask = mask.unsqueeze(1)
        mask = mask.expand(q.size(0), q.size(1), -1, -1)
        scores = scores.masked_fill(mask == 0, 1e-9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        # self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        # Cross Attention
        # x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        # src_mask))
        # x2 = self.norm_3(x)
        #######
        x = x + self.dropout_3(self.ff(x2))
        return x    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    # Need to combine the source and the target vocab
    def __init__(self, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        # remove the encoder
        # self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, trg, trg_mask):
        # e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, trg_mask)
        output = self.out(d_output)
        return output

def get_model(opt, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    # Modified the model parameters
    model = Transformer(trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
       
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model

####### Code for Assignment 
# Creating a dataset class
class WikiDataset(Dataset):
    def __init__(self, data, block_size):
        super(WikiDataset, self).__init__()
        self.block_size = block_size
        self.data = [data[i:i+block_size+1] for i in range(0, len(data), block_size)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = torch.tensor(self.data[index], dtype=torch.long)
        # All the text except the last token
        input_text = text[:-1]
        # All the text except the first token
        target_text = text[1:]
        return input_text, target_text
    
# Code for batching along with the padding    
def collate_fn(batch):
    input_text, target_text = zip(*batch)
    inputs = pad_sequence(input_text, batch_first=True, padding_value=0)
    # Padding value is set to be -100 so that the pad tokens are not considered 
    # in the gradient calculation 
    targets = pad_sequence(target_text, batch_first=True, padding_value=-100)
    return inputs, targets

def no_peak_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask

import matplotlib.pyplot as plt

def plot_metrics(training_losses, validation_losses, training_perplexities, validation_perplexities, filename='training_validation_metrics.png'):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_perplexities, label='Training Perplexity')
    plt.plot(validation_perplexities, label='Validation Perplexity')
    plt.title('Perplexity over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close the figure to free up memory
    print(f"Plot saved as {filename}")
#############################

def validate(model, valid_dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_text, target_text in valid_dataloader:
            input_text, target_text = input_text.to(device), target_text.to(device)
            input_mask = no_peak_mask(input_text.size(1)).to(device)
            output = model(input_text, input_mask)
            loss = loss_fn(output.view(-1, output.size(-1)), target_text.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(valid_dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity
    
def test(model, opt, test_dataloader, loss_fn):
    model.eval()
    device = opt.device
    total_loss = 0
    with torch.no_grad():
        for input_text, target_text in test_dataloader:
            input_text, target_text = input_text.to(device), target_text.to(device)
            input_mask = no_peak_mask(input_text.size(1)).to(device)
            output = model(input_text, input_mask)
            loss = loss_fn(output.view(-1, output.size(-1)), target_text.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(test_dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

    
def train_model(model, opt, train_dataloader, valid_dataloader, loss_fn):
    
    print("training model...")
    # write code to:
    #  1. create a nopeak mask
    #  2. feed training data to the model in batches
    #  3. send the indices of training tokens to the GPU
    #  4. linearize the predictions and compute the loss against ground truth
    #     (you can use F.cross_entropy or write your own code)
    #  5. calculate and apply the gradients with loss.backward() and optimizer.step()
    #  6. report intermediate trainining perplexity
    #  7. generate a test perplexity once per training epoch by calling test_model()
    #  8. save model weights to file specified in opt.savenam
    #  SEE trainer.py for examples of each of the above

    checkpoint_dir = os.path.join(opt.dir_name, 'checkpoints_part3')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Lists for storing metrics
    train_losses = []
    valid_losses = []
    train_perplexities = []
    valid_perplexities = []

    device = opt.device

    optimizer = opt.optimizer

    for epoch in range(opt.epochs):
        model.train()
        total_tl = 0

        for input_text, target_text in train_dataloader:
            input_text, target_text = input_text.to(device), target_text.to(device)
            input_mask = no_peak_mask(input_text.size(1)).to(device)
            output = model(input_text, input_mask)
            loss = loss_fn(output.view(-1, output.size(-1)), target_text.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_tl += loss.item()

        avg_tl = total_tl / len(train_dataloader)
        train_perplexity = math.exp(avg_tl)
        avg_vl, valid_perplexity = validate(model, valid_dataloader, loss_fn, device)

        train_losses.append(avg_tl)
        valid_losses.append(avg_vl)
        train_perplexities.append(train_perplexity)
        valid_perplexities.append(valid_perplexity)

        print(f"Epoch {epoch}: Train Loss {avg_tl:.4f}, Train Perplexity {train_perplexity:.4f}")
        print(f"Epoch {epoch}: Valid Loss {avg_vl:.4f}, Valid Perplexity {valid_perplexity:.4f}")

        if (epoch + 1) % 5 == 0:
            model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} at epoch {epoch+1}")

    return train_losses, train_perplexities, valid_losses, valid_perplexities



    
def test_model(model, opt, epoch):
    print("testing model...")
    model.eval()
    
    # write code to generate perplexity of test set
    
    model.train()

def main():
    
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=16)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)
                
    opt = parser.parse_args()
    opt.verbose = False    
    
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.device = torch.device("cuda:0")
    
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "//"
    opt.dir_name = dir_name
    shutil.copy(source_name,dir_name + source_name)
    opt.log_file = dir_name + "log_file.txt"
    
    print(str(opt))
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.train = read_corpus('wiki2.train.txt',tokenizer)
    opt.valid = read_corpus('wiki2.valid.txt',tokenizer)
    opt.test = read_corpus('wiki2.test.txt',tokenizer)
    
    obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()
    
    # Need a single vocab size
    model = get_model(opt,opt.vocab_size)
    model = model.to(opt.device)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    text = 'total params: %d' % (params)
    print(text)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0


    # Code goes here
    # Train Dataloader
    train_dataset = WikiDataset(opt.train, block_size=opt.seqlen)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, collate_fn=collate_fn)

    # Valid Dataloader
    valid_dataset = WikiDataset(opt.valid, block_size=opt.seqlen)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batchsize, shuffle=True, collate_fn=collate_fn)

    # Test Dataloader
    test_dataset = WikiDataset(opt.test, block_size=opt.seqlen)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=True, collate_fn=collate_fn)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            
    train_losses, train_perplexities, \
        valid_losses, valid_perplexities = train_model(model, opt, train_dataloader, \
                                                       valid_dataloader, loss_fn)
    
    # Plot the graphs
    plot_metrics(train_losses, valid_losses, train_perplexities, valid_perplexities)
    # test_model(model,opt,-1)
    avg_loss, perplexity = test(model, opt, test_dataloader, loss_fn)
    print(f"Test Loss {avg_loss:.4f}, Test Perplexity {perplexity:.4f}")
        
if __name__ == "__main__":
    main()        
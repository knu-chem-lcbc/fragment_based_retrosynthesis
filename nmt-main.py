#!/usr/bin/python3

import unicodedata
import re
import math
import psutil
import time
import datetime
from io import open
import random
from random import shuffle
import argparse
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.cuda

import sys; sys.argv=['']; del sys

#from tanimoto import *

use_cuda = torch.cuda.is_available()
print('is cuda available? ', use_cuda)

bidirectional = True
if bidirectional:
    directions = 2
else:
    directions = 1

SOS_token = 0
EOS_token = 1

# change dataset information accordingly

input_lang_name = 'PD2'
output_lang_name = 'RD2'
dataset = 'bi'
raw_data_file_path = ('PD2_RD2.txt',)

# from prod to reac
reverse=False

# how many prediction will be printed
n = 10

test_eval_every = 1
test_eval_score = 10
tics = 2
perc_train_set = 0.90
# create a text file.
create_txt = True

save_weights = True

# num of layers in both enc. and dec.
layers = 2
hidden_size = 2000
dropout = 0.1
# training set batch size.
batch_size = 16
# test set batch size.
test_batch_size = 8

epochs = 40
# initial lr
learning_rate= 4

#lr_schedule = {5:2, 10:5, 20:5, 40:2 }
lr_schedule = {4:1.17, 8:1.17, 12:1.17, 20:1.17, 24:1.17, 28:1.17, 32:1.17, 36:1.17}

criterion = nn.NLLLoss()


class Lang:
    def __init__(self, language):
        self.language_name = language
        self.word_to_index = {"SOS":SOS_token, "EOS":EOS_token}
        self.word_to_count = {}
        self.index_to_word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.vocab_size = 2
        self.cutoff_point = -1

    def countSentence(self, sentence):
        for word in sentence.split(' '):
            self.countWords(word)

    def countWords(self, word):
        if word not in self.word_to_count:
            self.word_to_count[word] = 1
        else:
            self.word_to_count[word] += 1

    def addSentence(self, sentence):
        new_sentence = ''
        for word in sentence.split(' '):
            unk_word = self.addWord(word)
            if not new_sentence:
                new_sentence =unk_word
            else:
                new_sentence = new_sentence + ' ' + unk_word
        return new_sentence

    def addWord(self, word):
        if self.word_to_count[word] > self.cutoff_point:
            if word not in self.word_to_index:
                self.word_to_index[word] = self.vocab_size
                self.index_to_word[self.vocab_size] = word
                self.vocab_size += 1
            return word
        else:
            return self.index_to_word[2]

def prepareLangs(lang1, lang2, file_path, reverse=False):
    print("Reading lines...")
    lines = open(file_path[0], encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]
    #print(type(pairs))
    #print(pairs)

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, file_path, reverse=False, perc_train_set=0.9, print_to=None):
    input_lang, output_lang, pairs = prepareLangs(lang1, lang2, file_path, reverse)
    print("Read %s sentence pairs" % len(pairs))

    if print_to:
        with open(print_to,'a') as f:
            f.write("Read %s sentence pairs \n" % len(pairs))


    print("Counting words...")
    for pair in pairs:
        input_lang.countSentence(pair[0])
        output_lang.countSentence(pair[1])


    pairs = [(input_lang.addSentence(pair[0]),output_lang.addSentence(pair[1])) for pair in pairs]

    shuffle(pairs)

    train_pairs = pairs[:math.ceil(perc_train_set*len(pairs))]
    #print('train pairs: ', train_pairs)
    test_pairs = pairs[math.ceil(perc_train_set*len(pairs)):]
    #print('test pairs: ', test_pairs)

    print("Train pairs: %s" % (len(train_pairs)))
    print("Test pairs: %s" % (len(test_pairs)))
    print("Counted Words -> Vocabulary Sizes (w/ EOS and SOS tags):")
    print("%s, %s -> %s" % (input_lang.language_name, len(input_lang.word_to_count),
                            input_lang.vocab_size,))
    print("%s, %s -> %s" % (output_lang.language_name, len(output_lang.word_to_count),
                            output_lang.vocab_size))
    print()

    if print_to:
        with open(print_to,'a') as f:
            f.write("Train pairs: %s" % (len(train_pairs)))
            f.write("Test pairs: %s" % (len(test_pairs)))
            f.write("Counted Words -> Vocabulary Sizes (w/ EOS and SOS tags):")
            f.write("%s, %s -> %s" % (input_lang.language_name,
                                      len(input_lang.word_to_count),
                                      input_lang.vocab_size,))
            f.write("%s, %s -> %s \n" % (output_lang.language_name, len(output_lang.word_to_count),
                            output_lang.vocab_size))

    return input_lang, output_lang, train_pairs, test_pairs

def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        indexes.append(lang.word_to_index[word])

    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1)
    if use_cuda:
        return result.cuda()
    else:
        return result

def tensorsFromPair(input_lang, output_lang, pair):
    input_variable = tensorFromSentence(input_lang, pair[0])
    target_variable = tensorFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


def sentenceFromTensor(lang, tensor):
    raw = tensor.data
    words = []
    for num in raw:
        words.append(lang.index_to_word[num.item()])
    return ' '.join(words)

def batchgen(data, input_lang, output_lang, batch_size, shuffle_data=True):
    if shuffle_data == True:
        shuffle(data)
    number_of_batches = len(data) // batch_size
    batches = list(range(number_of_batches))
    longest_elements = list(range(number_of_batches))

    for batch_number in range(number_of_batches):
        longest_input = 0
        longest_target = 0
        input_variables = list(range(batch_size))
        target_variables = list(range(batch_size))
        index = 0
        for pair in range((batch_number*batch_size),((batch_number+1)*batch_size)):
            input_variables[index], target_variables[index] = tensorsFromPair(input_lang, output_lang, data[pair])
            if len(input_variables[index]) >= longest_input:
                longest_input = len(input_variables[index])
            if len(target_variables[index]) >= longest_target:
                longest_target = len(target_variables[index])
            index += 1
        batches[batch_number] = (input_variables, target_variables)
        longest_elements[batch_number] = (longest_input, longest_target)
    return batches , longest_elements, number_of_batches


def pad_batch(batch):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(batch[0],padding_value=EOS_token)
    padded_targets = torch.nn.utils.rnn.pad_sequence(batch[1],padding_value=EOS_token)
    return (padded_inputs, padded_targets)

class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,layers=1,dropout=0.1, bidirectional=True):
        super(EncoderRNN, self).__init__()

        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(input_size,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=layers,dropout=dropout, bidirectional=bidirectional, batch_first=False)
        #self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size*self.directions, hidden_size)

    def forward(self, input_data, h_hidden, c_hidden):
        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))

        return hiddens, outputs

    def create_init_hiddens(self, batch_size):
        h_hidden = Variable(torch.zeros(self.num_layers*self.directions, batch_size, self.hidden_size))
        c_hidden = Variable(torch.zeros(self.num_layers*self.directions, batch_size, self.hidden_size))
        #h_hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        #c_hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if torch.cuda.is_available():
            return h_hidden.cuda(), c_hidden.cuda()
        else:
            return h_hidden, c_hidden

class DecoderAttn(nn.Module):
    def __init__(self, hidden_size, output_size, layers=1, dropout=0.1, bidirectional=True):
        super(DecoderAttn, self).__init__()

        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(output_size,hidden_size)
        self.dropout = nn.Dropout(dropout)
        #self.score_learner = nn.Linear(hidden_size, hidden_size)
        self.score_learner = nn.Linear(hidden_size*self.directions, hidden_size*self.directions)
        self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size, num_layers=layers,dropout=dropout,bidirectional=bidirectional, batch_first=False)
        #self.context_combiner = nn.Linear((hidden_size+hidden_size), hidden_size)
        self.context_combiner = nn.Linear((hidden_size*self.directions)+(hidden_size*self.directions), hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax(dim=1)
        self.log_soft = nn.LogSoftmax(dim=1)


    def forward(self, input_data, h_hidden, c_hidden, encoder_hiddens):

        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)
        batch_size = embedded_data.shape[1]
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))
        top_hidden = outputs[0].view(self.num_layers,self.directions, hiddens.shape[1], self.hidden_size)[self.num_layers-1]
        #top_hidden = outputs[0].view(self.num_layers, 1, hiddens.shape[1], self.hidden_size)[self.num_layers-1]
        top_hidden = top_hidden.permute(1,2,0).contiguous().view(batch_size,-1, 1)

        prep_scores = self.score_learner(encoder_hiddens.permute(1,0,2))
        scores = torch.bmm(prep_scores, top_hidden)
        attn_scores = self.soft(scores)
        con_mat = torch.bmm(encoder_hiddens.permute(1,2,0),attn_scores)
        h_tilde = self.tanh(self.context_combiner(torch.cat((con_mat, top_hidden),dim=1).view(batch_size,-1)))
        pred = self.output(h_tilde)
        pred = self.log_soft(pred)

        return pred, outputs


def train_batch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_batch.shape[1])

    enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)

    decoder_input = Variable(torch.LongTensor(1,input_batch.shape[1]).fill_(output_lang.word_to_index.get("SOS")).cuda()) if use_cuda else Variable(torch.LongTensor(1,input_batch.shape[1]).fill_(output_lang.word_to_index.get("SOS")))

    dec_h_hidden = enc_outputs[0]
    dec_c_hidden = enc_outputs[1]

    for i in range(target_batch.shape[0]):
        pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

        decoder_input = target_batch[i].view(1,-1)
        dec_h_hidden = dec_outputs[0]
        dec_c_hidden = dec_outputs[1]

        loss += loss_criterion(pred,target_batch[i])


    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(),args.clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(),args.clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_batch.shape[0]

def train(train_batches, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion):

    round_loss = 0
    i = 1
    for batch in train_batches:
        i += 1
        (input_batch, target_batch) = pad_batch(batch)
        batch_loss = train_batch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion)
        round_loss += batch_loss

    return round_loss / len(train_batches)

def test_batch(input_batch, target_batch, encoder, decoder, loss_criterion):

    loss = 0

    #create initial hidde state for encoder
    enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_batch.shape[1])

    enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)

    decoder_input = Variable(torch.LongTensor(1,input_batch.shape[1]).fill_(output_lang.word_to_index.get("SOS")).cuda()) if use_cuda else Variable(torch.LongTensor(1,input_batch.shape[1]).fill_(output_lang.word_to_index.get("SOS")))
    dec_h_hidden = enc_outputs[0]
    dec_c_hidden = enc_outputs[1]

    for i in range(target_batch.shape[0]):
        pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

        topv, topi = pred.topk(1,dim=1)
        ni = topi.view(1,-1)

        decoder_input = ni
        dec_h_hidden = dec_outputs[0]
        dec_c_hidden = dec_outputs[1]

        loss += loss_criterion(pred,target_batch[i])

    return loss.item() / target_batch.shape[0]

def test(test_batches, encoder, decoder, loss_criterion):

    with torch.no_grad():
        test_loss = 0

        for batch in test_batches:
            (input_batch, target_batch) = pad_batch(batch)
            batch_loss = test_batch(input_batch, target_batch, encoder, decoder, loss_criterion)
            test_loss += batch_loss

    return test_loss / len(test_batches)


def evaluate(encoder, decoder, sentence, cutoff_length):
    with torch.no_grad():
        input_variable = tensorFromSentence(input_lang, sentence)
        input_variable = input_variable.view(-1,1)
        enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(1)

        enc_hiddens, enc_outputs = encoder(input_variable, enc_h_hidden, enc_c_hidden)

        decoder_input = Variable(torch.LongTensor(1,1).fill_(output_lang.word_to_index.get("SOS")).cuda()) if use_cuda else Variable(torch.LongTensor(1,1).fill_(output_lang.word_to_index.get("SOS")))
        dec_h_hidden = enc_outputs[0]
        dec_c_hidden = enc_outputs[1]

        decoded_words = []

        for di in range(cutoff_length):
            pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)

            topv, topi = pred.topk(1,dim=1)
            ni = topi.item()
            if ni == output_lang.word_to_index.get("EOS"):
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index_to_word[ni])

            decoder_input = Variable(torch.LongTensor(1,1).fill_(ni).cuda()) if use_cuda else Variable(torch.LongTensor(1,1).fill_(ni))
            dec_h_hidden = dec_outputs[0]
            dec_c_hidden = dec_outputs[1]

        output_sentence = ' '.join(decoded_words)

        return output_sentence

def molGen(input):
    Slist = list()
    input.strip()
    if '-' in input:
        input = input.split('-')
        R1 = input[0].split()
        R2 = input[1].split()
        Slist.append(R1)
        Slist.append(R2)
    else:
        R = input.split()
        Slist.append(R)
    return Slist

def tanimoto(truth, prediction, i, j):
    return len(set(truth[i]) & set(prediction[j])) / float(len(set(truth[i]) | set(prediction[j])))

def molGenX(input):
    Slist = list()
    input.strip()
    if '-' in input:
        input = input.split('-')
        R1 = input[0].split()
        R2 = input[1].split()[:-1]
        Slist.append(R1)
        Slist.append(R2)
    else:
        R = input.split()[:-1]
        Slist.append(R)
    return Slist


def similarity(truth, prediction):
    # Sdict = Similarity dictiontionary, Nlist = NameList, Vlist = Value list
    Sdict = dict()
    if len(truth) == 2 and len(prediction) == 2:
        # ground truth A >> B + C. Prediction A >> D + E
        Nlist = ['DB', 'DC', 'EB', 'EC']
        Vlist = [(0,0), (1,0), (0,1), (1,1)]

        for i in range(4):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['DB'] >= Sdict['DC']:
            del Sdict['DC']
        else:
            del Sdict['DB']

        if Sdict['EB'] >= Sdict['EC']:
            del Sdict['EC']
        else:
            del Sdict['EB']

    # Condition 2

    elif len(truth) == 1 and len(prediction) == 2:
        # ground truth A >> G. Prediction A >> D + E
        Nlist = ['DG', 'EG']
        Vlist = [(0,0), (0,1)]

        for i in range(2):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['DG'] >= Sdict['EG']:
            del Sdict['EG']
        else:
            del Sdict['DG']

    # Condition 3

    elif len(truth) == 2 and len(prediction) == 1:
        # ground truth A >> B + C. Prediction A >> F
        Nlist = ['FB', 'FC']
        Vlist = [(0,0), (1,0)]

        for i in range(2):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['FB'] >= Sdict['FC']:
            del Sdict['FC']
        else:
            del Sdict['FB']

    # Condition 4

    elif len(truth) == 1 and len(prediction) == 1:
        # ground truth A >> G. Prediction A >> F
        Nlist = ['FG']
        Vlist = [(0,0)]
        Sdict[Nlist[0]] = tanimoto(truth, prediction, Vlist[0][0], Vlist[0][1])

    else:
        Sdict['Prediction'] = 0


    return Sdict



def evaluate_randomly(encoder, decoder, pairs, n=2):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_sentence = evaluate(encoder, decoder, pair[0],cutoff_length=90)
        print('<', output_sentence)
        print('test me if you can')
        print('')
        if create_txt:
            f = open(print_to, 'a')
            f.write("\n \
                > %s \n \
                = %s \n \
                < %s \n" % (pair[0], pair[1], output_sentence))
            f.close()


def evaluate_sequentially(encoder, decoder, pairs):
    scores = list()
    for pair in pairs:
        truth = molGen(pair[1])
        prediction = evaluate(encoder, decoder, pair[0],cutoff_length=90)
        prediction = molGenX(prediction)
        Sdict = similarity(truth, prediction)
        for item in Sdict.items():
            scores.append(item)

    sum = 0; count = 0; thresh = 0
    bad = 0; zeros = 0; seven = 0; five = 0

    for i in range(len(scores)):
        sum += scores[i][1]
        if scores[i][1] == 1:
            print('exacts', scores[i])
            print('')
            count += 1
        elif scores[i][1] >= 0.85:
            print('goods', scores[i])
            print('')
            thresh += 1
        elif scores[i][1] == 0:
            print('zeros', scores[i])
            print('')
            zeros += 1
        elif scores[i][1] >= 0.70:
            print('sevens', scores[i])
            print('')
            seven += 1
        elif scores[i][1] >= 0.50:
            print('fives', scores[i])
            print('')
            five += 1
        else:
            print('bads', scores[i])
            print('')
            bad += 1
    print('exact-goods-zeros-bads-sevens-fives', count, thresh, zeros, bad, seven, five)
    ave = sum/len(scores)
    exact_match = count
    similar = thresh
    bads = bad
    print('')
    if create_txt:
        f = open(print_to, 'a')
        f.write("\n \
            Average similarity metric %s \n \
            Number of exact match %s \n \
            Number of similar pairs %s \n \
            Number of zeros %s \n \
            Number of sevens %s \n \
            Number of fives %s \n \
            Number of bads %s \n" % (ave, exact_match, similar, zeros, seven, five, bads))
        f.close()


def mem():
    if use_cuda:
        mem = torch.cuda.memory_allocated()/1e7
    else:
        mem = psutil.cpu_percent()
    print('Current mem usage:')
    print(mem)
    return "Current mem usage: %s \n" % (mem)



def asHours(s):
    m = math.floor(s / 60)
    h = math.floor(m / 60)
    s -= m * 60
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def train_and_test(epochs, test_eval_every, tics, learning_rate, lr_schedule, train_pairs, test_pairs, input_lang, output_lang, batch_size, test_batch_size, encoder, decoder, loss_criterion, save_weights):

    times = []
    losses = {'train set':[], 'test set': []}

    test_batches, longest_seq, n_o_b = batchgen(test_pairs, input_lang, output_lang, test_batch_size, shuffle_data=False)

    start = time.time()
    for i in range(1,epochs+1):

        if i in lr_schedule.keys():
            learning_rate /= lr_schedule.get(i)


        encoder.train()
        decoder.train()

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        batches, longest_seq, n_o_b = batchgen(train_pairs, input_lang,
                                           output_lang, batch_size,
                                           shuffle_data=True)
        train_loss = train(batches, encoder, decoder, encoder_optimizer,
                       decoder_optimizer, loss_criterion)

        now = time.time()
        print("Iter: %s \nLearning Rate: %s \nTime: %s \nTrain Loss: %s \n" % (i, learning_rate, asHours(now-start), train_loss))
        mem()
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

        if create_txt:
            with open(print_to, 'a') as f:
                f.write("Iter: %s \nLeaning Rate: %s \nTime: %s \nTrain Loss: %s \n" % (i, learning_rate, asHours(now-start), train_loss))

        if i % test_eval_every == 0:
            if test_pairs:
                test_loss = test(test_batches, encoder, decoder, criterion)
                print("Test set loss: %s" % (test_loss))
                if create_txt:
                    with open(print_to, 'a') as f:
                        f.write("Test Loss: %s \n" % (test_loss))
                evaluate_randomly(encoder, decoder, test_pairs, n)
                evaluate_sequentially(encoder, decoder, test_pairs)
            else:
                evaluate_randomly(encoder, decoder, train_pairs, n)



        if i % tics == 0:
            times.append((time.time()-start)/60)
            losses['train set'].append(train_loss)
            if test_pairs:
                losses['test set'].append(test_loss)
            if save_weights:
                torch.save(encoder.state_dict(), output_file_name+'_enc_weights.pt')
                torch.save(decoder.state_dict(), output_file_name+'_dec_weights.pt')


use_cuda = torch.cuda.is_available()


output_file_name = "testdata.%s_hidden.%s_dropout.%s_learningrate.%s_batch.%s_epochs.%s" % (dataset,hidden_size,dropout,learning_rate,batch_size,epochs)

if create_txt:
    print_to = output_file_name+'.txt'
    with open(print_to, 'w+') as f:
        f.write("Starting Training \n")
else:
    print_to = None

input_lang, output_lang, train_pairs, test_pairs = prepareData(input_lang_name, output_lang_name, raw_data_file_path,reverse=reverse, perc_train_set=perc_train_set, print_to=print_to)
print('Train Pairs #')
print(len(train_pairs))


parser = argparse.ArgumentParser(description='Fragment Model')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
args = parser.parse_args()


mem()

if create_txt:
    with open(print_to, 'a') as f:
        f.write("\nRandom Train Pair: %s \n\nRandom Test Pair: %s \n\n" % (random.choice(train_pairs),random.choice(test_pairs) if test_pairs else "None"))
        f.write(mem())


#create the Encoder
encoder = EncoderRNN(input_lang.vocab_size, hidden_size, layers=layers,
                     dropout=dropout, bidirectional=bidirectional)

#create the Decoder
decoder = DecoderAttn(hidden_size, output_lang.vocab_size, layers=layers,
                      dropout=dropout, bidirectional=bidirectional)

print('Encoder and Decoder Created')
mem()

if use_cuda:
    print('Cuda being used')
    encoder = encoder.cuda()
    decoder = decoder.cuda()

print('Number of epochs: '+str(epochs))

if create_txt:
    with open(print_to, 'a') as f:
        f.write('Encoder and Decoder Created\n')
        f.write(mem())
        f.write("Number of epochs %s \n" % (epochs))

train_and_test(epochs, test_eval_every, tics, learning_rate, lr_schedule, train_pairs, test_pairs, input_lang, output_lang, batch_size, test_batch_size, encoder, decoder, criterion, save_weights)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


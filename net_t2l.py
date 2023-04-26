from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn as nn
import torch

from net import Net
from data import T2LData, load_intent_data
from finetuning import T2LFineTuner

class T2LModel(nn.Module):

    def __init__(self, inp_shape, hidden_shape, out_shape, transfer='sigmoid'):
        super(T2LModel, self).__init__()
        self.inp_shape = inp_shape
        self.hidden_shape = hidden_shape
        self.out_shape = out_shape

        self.transfer = torch.sigmoid if transfer == 'sigmoid' else torch.tanh
        self.input_layer = nn.Linear(self.inp_shape, self.hidden_shape[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_shape[hi-1], self.hidden_shape[hi]) for hi in range(1, len(self.hidden_shape))])
        self.output_layer = nn.Linear(self.hidden_shape[-1], self.out_shape)

        self.train()

    def forward(self, x):
        x = self.transfer(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.transfer(layer(x))
        
        return self.output_layer(x)
    
    def new_m(self, m, keep_weights=True):
        if keep_weights:
            w, b = self.output_layer.weight, self.output_layer.bias
            self.output_layer = nn.Linear(self.hidden_shape[-1], m)
            self.output_layer.weight.data[:w.size(0), :w.size(1)] = w.data
            self.output_layer.bias.data[:b.size(0)] = b.data
        else:
            self.output_layer = nn.Linear(self.hidden_shape[-1], m)

class NetT2L(Net):
    """ Text 2 Label """

    def __init__(self, hidden=[50, 75]):
        Net.__init__(self)
        
        self.text2vec = SentenceTransformer('fav-kky/FERNET-C5')
        
        self.data = T2LData(net=self, data=load_intent_data(from_zero=True))
        self.trainer = T2LFineTuner(net=self)

        # Model
        self.hidden = hidden
        self.model = T2LModel(self.data.n, self.hidden, self.data.m)
        self.trainer.reinit_optimizer()

    def encode(self, x):
        return self.text2vec.encode(x)
    
    def decode(self, v):
        return self.data.target2label[torch.argmax(v).item()]
    
    def reinit_model(self, keep_weights=True):
        self.model.new_m(self.data.m, keep_weights=keep_weights)
        self.trainer.reinit_optimizer()

    def predict(self, x, is_encoded=False):
        self.model.eval()
        return self.decode(self.model(x if is_encoded else torch.from_numpy(self.encode(x))))
    
    def learn(self, sample, label, verbose=False):
        self.data.add(sample, label)
        self.trainer.fit(trainloader=self.data.loader(group='train'), devloader=self.data.loader(group='dev'), verbose=verbose)

    def evaluate(self, ret_wrongs=False, ret_oks=False, verbose=False):
        self.model.eval()
        loss_list = []
        oks = []
        wrongs = []
        n_correct = 0
        n_fail = 0
        for x, y_true, sample, label in self.data.loader(group='knowledge', batch_size=1):
            
            y_pred = self.model(x)
            loss_list.append(self.trainer.criterion(y_pred, y_true).data)
            
            target_pred = torch.argmax(y_pred).item()
            if target_pred == y_true[0].item():
                n_correct += 1
                oks.append((sample[0], label[0]))
            else:
                n_fail += 1
                wrongs.append((sample[0], label[0], self.data.target2label[target_pred]))
        
        acc = n_correct / (n_correct + n_fail)
        loss = np.mean([l.item() for l in loss_list])
        
        if verbose:
            print(f'Loss: {loss}, Acc: {acc}')
        
        ret = [loss, acc]
        
        if ret_wrongs:
            ret.append(wrongs)
        
        if ret_oks:
            ret.append(oks)
        
        return ret

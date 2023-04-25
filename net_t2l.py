from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn as nn
import torch

from net import Net
from data import T2LData
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
    
    def new_m(self, m):
        self.output_layer = nn.Linear(self.hidden_shape[-1], m)


class NetT2L(Net):
    """ Text 2 Label """

    def __init__(self, hidden=[50, 75]):
        Net.__init__(self)
        
        self.text2vec = SentenceTransformer('fav-kky/FERNET-C5')
        
        self.data_train = T2LData(net=self, group='train', samples=['Ahoj.', 'Najdi mi tohle.'], labels=['hi', 'search'])
        self.data_dev = T2LData(net=self, group='dev', samples=['Ahoj.', 'Najdi mi tohle.'], labels=['hi', 'search'])
        self.trainer = T2LFineTuner(net=self)

        # Model
        self.hidden = hidden
        self.model = T2LModel(self.data_train.n, self.hidden, self.data_train.m)
        self.trainer.reinit_optimizer()

    def encode(self, x):
        return self.text2vec.encode(x)
    
    def decode(self, v):
        return self.data_train.target2label[torch.argmax(v).item()]
    
    def reinit_model(self, keep_weights=True, print_summary=True):
        self.model.new_m(self.data_train.m)
        self.trainer.reinit_optimizer()

    def predict(self, x, is_encoded=False):
        self.model.eval()
        return self.decode(self.model(x if is_encoded else torch.from_numpy(self.encode(x))))
    
    def learn(self, samples, labels, reset=False, verbose=False):
        # TODO if label not in labels...

        if reset:
            self.data_train.set_pairs(samples, labels)
            self.data_dev.set_pairs(samples, labels)
        else:
            self.data_train.set_pairs(self.data_train.samples+self.data_dev.samples+samples, self.data_train.labels+self.data_dev.labels+labels)
            self.data_dev.set_pairs(samples, labels)

        self.trainer.fit(trainloader=self.data_train.loader(), devloader=self.data_dev.loader(), verbose=verbose)

    def evaluate(self):
        self.model.eval()
        loss_list = []
        n_correct = 0
        n_fail = 0
        for x, y_true, _, _ in self.data_dev.loader():
            
            y_pred = self.model(x)
            loss_list.append(self.criterion(y_pred, y_true).data)
            
            if torch.argmax(y_pred).item() == y_true[0].item():
                n_correct += 1
            else:
                n_fail += 1
        
        acc = n_correct / (n_correct + n_fail)
        loss = np.mean([l.item() for l in loss_list])
        print(f'Loss: {loss}, Acc: {acc}')
        
        return loss, acc

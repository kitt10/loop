from torchvision import models
import numpy as np
import torch.nn as nn
import torch

from net import Net
from data_i2l import I2LData, path2vec, load_image_data
from finetuning import I2LFineTuner

class NetI2L(Net):
    """ Image 2 Label """

    def __init__(self):
        Net.__init__(self)

        self.path2vec = path2vec
        
        self.data = I2LData(net=self, data=load_image_data('i2l-dataset'))
        self.trainer = I2LFineTuner(net=self)

        # Model
        self.model = models.detection.ssdlite320_mobilenet_v3_large(
            num_classes=self.data.m,
            pretrained=True,
            pretrained_backbone=True, 
            trainable_backbone_layers=0
        )
        
        self.trainer.reinit_optimizer(parameters=self.model.parameters())

    def encode(self, x):
        return self.path2vec(x)
    
    def decode(self, v):
        return self.data.target2label[torch.argmax(v).item()]
    
    def new_m(self, m, keep_weights=False):
        if keep_weights:
            # TODO
            pass
        else:
            self.model = models.detection.ssdlite320_mobilenet_v3_large(
                num_classes=self.data.m,
                pretrained=True,
                pretrained_backbone=True, 
                trainable_backbone_layers=0
            )
    
    def reinit_model(self, keep_weights=True):
        self.new_m(self.data.m, keep_weights=keep_weights)
        self.trainer.reinit_optimizer(parameters=self.model.parameters())

    def predict(self, x, is_encoded=False):
        self.model.eval()
        
        
        print(self.model(x if is_encoded else torch.stack([self.encode(x)])))


        return self.decode(self.model(x if is_encoded else torch.stack([self.encode(x)])))
    
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

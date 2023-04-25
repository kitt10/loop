from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import torch

class T2LDataset(Dataset):

    def __init__(self, x, y, samples, labels):
        self.x = x
        self.y = y
        self.samples = samples
        self.labels = labels       

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.samples[index], self.labels[index]

    def __len__(self):
        return self.x.shape[0]
    

class T2LData:

    def __init__(self, net, group, samples, labels, batch_size=16, shuffle=True):
        """
            samples: list: ['sentence 1', 'sentence 2', ... 'sentence n']
            labels: list: ['label 1', 'label 2', ... 'label n']
            encoder: def for encoding input (in: <string> sentence, out: <np.ndarray> vec)
            decoder: def for decoding output (in: <np.ndarray> vec, <string[]> labels, out: <string> label)
        """

        self.net = net
        self.group = group
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.set_pairs(self.samples, self.labels)

    def set_pairs(self, samples, labels, extend=False):
        if extend:
            self.samples.extend(samples)
            self.labels.extend(labels)
        else:
            self.samples = samples
            self.labels = labels

        self.samples_encoded = np.array([self.net.encode(sample) for sample in self.samples])
        self.sorted_labels = sorted(list(set(self.labels)))
        self.target2label = {target:label for target, label in enumerate(self.sorted_labels)}
        self.label2target = [self.sorted_labels.index(label) for label in self.labels]

        self.x = torch.from_numpy(self.samples_encoded.reshape(-1, self.samples_encoded.shape[1]).astype('float32'))
        self.y = torch.tensor(self.label2target)
        self.y_one_hot = F.one_hot(self.y)
        
        self.p = self.x.shape[0]
        self.n = len(self.x[0])
        self.m = len(self.sorted_labels)

        print(f'Data {self.group}: n = {self.n}, m = {self.m}, p = {self.p}')
        print(f'Labels: {self.target2label}')
    
    def loader(self):
        return DataLoader(dataset=T2LDataset(
            x=self.x,
            y=self.y,
            samples=self.samples,
            labels=self.labels
        ), batch_size=self.batch_size, shuffle=self.shuffle)
    
    def label(self, one_hot):
        return self.net.decode(one_hot)

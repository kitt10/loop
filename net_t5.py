from transformers import T5Tokenizer, T5ForConditionalGeneration

from net import Net
from data import Data
from finetuning import T5FineTuner


class NetT5(Net):

    def __init__(self, huggingname='t5-small', name='t5'):
        Net.__init__(self, name=name)

        self.huggingname = huggingname

        self.tokenizer = T5Tokenizer.from_pretrained(self.huggingname)
        self.model = T5ForConditionalGeneration.from_pretrained(self.huggingname)
        
        self.data = Data(net=self)
        self.trainer = T5FineTuner(net=self)

    def encode(self, x, ten='pt'):
        return self.tokenizer(x, return_tensors=ten).input_ids
    
    def decode(self, v):
        return self.tokenizer.decode(v, skip_special_tokens=True)

    def predict(self, x):
        input_ids = self.encode(x)
        outputs = self.model.generate(input_ids)
        return self.decode(outputs[0])
    
    def learn(self, pairs, reset=False):
        trainloader, devloader = self.data.make_loaders(pairs, reset)
        self.trainer.fit(trainloader, devloader)

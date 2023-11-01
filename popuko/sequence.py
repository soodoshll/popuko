from typing import Optional, Union
import transformers
import torch

class TokenSeq:
    def __init__(self, model, tokenizer, token_ids:Union[str, list], kvcache = None):
        self.model = model
        self.tokenizer = tokenizer
        if isinstance(token_ids, str):
            token_ids = self.tokenizer(token_ids, return_tensors="pt").input_ids[0]
        self.token_ids = token_ids
        self.kvcache = kvcache

    def __getitem__(self, index):
        if isinstance(index, int):
            index = slice(index, index + 1)
        if not isinstance(index, slice):
            raise ValueError()
        start, stop, step = index.start, index.stop, index.step
        if stop is None:
            stop = len(self.token_ids) - 1
        if start is None:
            start = 0
        assert start >= 0
        assert stop < len(self.token_ids)
        if step != None:
            raise ValueError()
        if start == 0 and self.kvcache is not None:
            kv = [[t[:, :, :stop] for t in layer_kv] for layer_kv in self.kvcache]
        else:
            kv = None
        return TokenSeq(self.model, self.tokenizer, self.token_ids[start:stop:step], kv)
    
    def __str__(self):
        return self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
    
    def generate(self, max_new_tokens=20):
        if self.kvcache is None:
            kvcache_len = 0 
            input_ids = self.token_ids
        else:
            kvcache_len = self.kvcache[0][0].shape[2]
            input_ids = self.token_ids[kvcache_len:]
        if input_ids.shape[0] == 0:
            input_ids = self.token_ids[-1:]
            self.kvcache = [[t[:, :, :-1] for t in layer_kv] for layer_kv in self.kvcache]
            kvcache_len = self.kvcache[0][0].shape[2]
        
        # argmax sampling
        cnt = 0
        while cnt < max_new_tokens and input_ids[-1] != self.tokenizer.eos_token_id:
            cnt += 1
            out = self.model(input_ids, past_key_values=self.kvcache, use_cache=True)
            past_key_values = out.past_key_values
            assert past_key_values[0][0].shape[2] == kvcache_len + input_ids.shape[0], past_key_values[0][0].shape
            input_ids = out.logits[-1:].argmax(dim=1)
            self.kvcache = past_key_values
            kvcache_len = self.kvcache[0][0].shape[2]
            self.token_ids = torch.cat([self.token_ids, input_ids])
        return TokenSeq(model, tokenizer, self.token_ids, self.kvcache)
        
model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')


s = TokenSeq(model, tokenizer, "The university of Toronto is in the city of Toronto")
s1 = s.generate()
print(s1)

s2 = s[:5]
print(s2)

s3 = s2.generate()
print(s3)

s4 = TokenSeq(model, tokenizer, "The university of Toronto is") 
print(s4)
print(s4.generate())
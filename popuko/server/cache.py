from typing import List, Optional
import transformers
import torch

class Node:
    def __init__(self,
                 token_ids: List[int],
                 kv_cache: torch.Tensor, 
                 parent: Optional['Node'] = None):
        self.token_ids = token_ids
        self.kv_cache = kv_cache
        self.parent = parent
        self.children = []

class TrieCache:
    def __init__(self):
        self.root : List[Node] = []

    def match(self, seq: List[int]) -> Optional[Node]:
        def _match(node : Node, seq) -> Optional[Node]:
            seq_len = len(node.token_ids)
            if len(seq) <= seq_len:
                if seq == node.token_ids[:len(seq)]:
                    return node
                else:
                    return None
            if seq[:seq_len] == node.token_ids:
                tail = seq[seq_len:]
                for child in self.children:
                    ret = _match(child, tail)
                    if ret is not None:
                        return ret
            return None
        for child in self.root:
            ret = _match(child, seq)
            if ret is not None:
                return ret
        return None

class CachedTransformer:
    def __init__(self, model, tokenizer = None):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str | List[int]):
        if isinstance(prompt, str):
            prompt = self.tokenizer(prompt)
        pass

if __name__ == '__main__':
    model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    
    cached_model = CachedTransformer(model, tokenizer)
    cached_model.generate("I say it once")
    cached_model.generate("I say it once again")
    cached_model.generate("I say it once again and again")
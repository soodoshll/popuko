from typing import List, Optional, Tuple
import transformers
import torch

class Node:
    def __init__(self,
                 token_ids: List[int],
                 kv_cache: torch.Tensor, 
                #  logits: torch.Tensor,
                 parent: Optional['Node'] = None):
        self.token_ids = token_ids
        self.kv_cache = kv_cache
        # self.logits = logits
        self.parent = parent
        self.children = []

def index_kvcache(kv_cache, start, end):
    return [[kv[:, start:end].clone() for kv in layer] for layer in kv_cache]

def concat_kvcache(kv_cache_list):
    assert len(kv_cache_list) > 0
    # concat kv cache
    kv_tensors = []
    for layer in range(len(kv_cache_list[0])):
        layer_kv = []
        for k_or_v in range(2):
            kv = [kv_cache[layer][k_or_v] for kv_cache in kv_cache_list]
            # print(kv)
            kv = torch.cat(kv, dim=1)
            layer_kv.append(kv)
        kv_tensors.append(layer_kv)
    return kv_tensors


class TrieCache:
    def __init__(self):
        self.root = Node([], None, None)

    def lookup(self, seq: List[int], partial=True) -> Tuple[Optional[Node], List[torch.Tensor]]:
        def _lookup(node : Node, seq, partial) -> Tuple[Optional[Node], List[torch.Tensor]]:
            node_seq_len = len(node.token_ids)
            # print(node, seq)
            if partial and len(seq) <= node_seq_len:
                if seq == node.token_ids[:len(seq)]:
                    return node, [index_kvcache(node.kv_cache, 0, len(seq))]
                else:
                    return None, []
            if seq[:node_seq_len] == node.token_ids:
                tail = seq[node_seq_len:]
                node_kv = [] if node == self.root else [node.kv_cache]
                child_kv = []
                ret = self.root
                for child in node.children:
                    ret, child_kv = _lookup(child, tail, partial)
                    if ret is not None:
                        return ret, node_kv + child_kv
                return node, node_kv
            return None, []
        return _lookup(self.root, seq, partial) 

    def path_len(self, node: None) -> int:
        path_len = 0
        while node is not None:
            path_len += len(node.token_ids)
            node = node.parent
        return path_len

    def insert(self, token_ids, kv_cache):
        insert_point, _ = self.lookup(token_ids, partial=False)
        prefix_len = self.path_len(insert_point)
        if prefix_len == len(token_ids):
            return
        new_node = Node(token_ids[prefix_len:], index_kvcache(kv_cache, -prefix_len, None), insert_point)
        insert_point.children.append(new_node)


if __name__ == '__main__':
    model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
from typing import List, Optional, Tuple, Sequence
import transformers
import torch


class KVCache:
    def __init__(self, kv_list: Tuple[Tuple[torch.Tensor]]):
        assert len(kv_list) > 0
        assert len(kv_list[0]) == 2
        assert len(kv_list[0][0].shape) in (3, 4)
        self._kv_list: Tuple[Tuple[torch.Tensor]] = kv_list
        # [Layer, kv, batchsize (optional), num_heads, seqlen, dim]

    def is_batched(self):
        return len(self._kv_list[0][0].shape) == 4

    def __getitem__(self, key: int | slice | Sequence[int | slice]):
        """
        Index by seqlen or (batch, seqlen)
        """
        if isinstance(key, Sequence):
            assert self.is_batched()
            assert len(key) == 2
            batch_idx, key = key
            return KVCache([[kv[batch_idx, :, key].clone() for kv in layer] for layer in self._kv_list])
        else:
            if self.is_batched():
                return KVCache([[kv[:, :, key].clone() for kv in layer] for layer in self._kv_list])
            else:
                return KVCache([[kv[:, key].clone() for kv in layer] for layer in self._kv_list])

    @staticmethod
    def cat(kv_cache_list: Sequence["KVCache"]):
        assert len(kv_cache_list) > 0
        # concat kv cache
        kv_tensors = []
        for layer in range(kv_cache_list[0].nlayers()):
            layer_kv = []
            for k_or_v in range(2):
                kv = [kv_cache.hf()[layer][k_or_v] for kv_cache in kv_cache_list]
                kv = torch.cat(kv, dim=-2)
                layer_kv.append(kv)
            kv_tensors.append(layer_kv)
        return KVCache(kv_tensors)

    def hf(self):
        return self._kv_list

    def seqlen(self):
        return self._kv_list[0][0].shape[-2]

    def nlayers(self):
        return len(self._kv_list)


class TrieNode:
    def __init__(self, token_ids: List[int], kv_cache: torch.Tensor, parent: Optional["TrieNode"] = None):
        self.token_ids: List[int] = token_ids
        self.kv_cache: KVCache = kv_cache
        self.parent: Optional["TrieNode"] = parent
        self.children: List["TrieNode"] = []


class TrieCache:
    def __init__(self):
        self.root: TrieNode = TrieNode([], None, None)

    def lookup(self, seq: List[int], partial: bool = True) -> Tuple[Optional[TrieNode], List[torch.Tensor]]:
        def _lookup(node: TrieNode, seq: List[int], partial: bool) -> Tuple[Optional[TrieNode], List[torch.Tensor]]:
            node_seq_len = len(node.token_ids)
            if partial and len(seq) <= node_seq_len:
                if seq == node.token_ids[: len(seq)]:
                    return node, [node.kv_cache[0 : len(seq)]]
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

    def path_len(self, node: TrieNode) -> int:
        path_len = 0
        while node is not None:
            path_len += len(node.token_ids)
            node = node.parent
        return path_len

    def insert(self, token_ids: List[int], kv_cache: KVCache):
        insert_point, _ = self.lookup(token_ids, partial=False)
        prefix_len = self.path_len(insert_point)
        if prefix_len == len(token_ids):
            return
        new_node = TrieNode(token_ids[prefix_len:], kv_cache[-prefix_len:], insert_point)
        insert_point.children.append(new_node)

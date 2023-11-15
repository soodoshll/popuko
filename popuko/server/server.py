# A Simple LLM server supporting dynamic batching and
# exposing the control of kv-cache to users

from typing import Optional, List, Tuple, Sequence, Set

import threading
import queue
import time
import asyncio
import argparse

import transformers
import torch

from .cache import KVCache, TrieCache, TrieNode

DEFAULT_SERVER = None
DEFAULT_TOKENIZER = None

class Request:
    def __init__(
        self, 
        prompt: List[int], 
        max_new_tokens: Optional[int] = None, 
        kv_cache: Optional[KVCache] = None, 
        eos_token_id: Optional[int | Set[int]]=None,
        temperature: float = 0,
    ):
        # Original request info
        # Should keep constant
        self.prompt: List[int] = prompt
        self.kv_cache: Optional[KVCache] = kv_cache
        self.max_new_tokens: Optional[int] = max_new_tokens
        self.eos_token_id = eos_token_id
        self.temperature = temperature

        # Runtime info
        self._generated_tokens: List[int] = []
        self._event: threading.Event = threading.Event()
        self._trie_node: Optional[TrieNode] = None

    def add_token(self, token: int) -> None:
        self._generated_tokens.append(token)

    def finished(self) -> bool:
        return len(self._generated_tokens) >= self.max_new_tokens or self._generated_tokens[-1] == self.eos_token_id

    def update_kv_cache(self, cache: KVCache) -> None:
        self.kv_cache = cache

    def kv_len(self) -> int:
        if self.kv_cache is None:
            return 0
        else:
            return self.kv_cache.seqlen()

    def input_ids_len(self) -> int:
        return len(self.prompt) + len(self._generated_tokens) - self.kv_len()

    def input_ids(self) -> List[int]:
        return (self.prompt + self._generated_tokens)[self.kv_len() :]

    def tokens(self) -> List[int]:
        return self.prompt + self._generated_tokens
    
    def new_tokens(self) -> List[int]:
        return self._generated_tokens

    def has_cache(self) -> bool:
        return self.kv_cache is not None


class PopukoServer:
    def __init__(self, model, max_batch_size: int = 4, device: str | torch.device = "cuda", use_cache: bool = True):
        self.model = model
        self.queue: queue.Queue = queue.Queue()
        self.max_batch_size: int = max_batch_size
        self.finished: bool = False

        self.trie_cache: TrieCache = TrieCache()

        self.device: str | torch.device = device
        self.use_cache: bool = use_cache

        self.build_kv_time: float = 0
        self.server_thread: Optional[threading.Thread] = None

    def launch(self):
        self.server_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.server_thread.start()

    def _find_kv_cache(self, req: Request):
        # look up kv cache in the trie
        _, kv_cache = self.trie_cache.lookup(req.tokens())
        if len(kv_cache) > 0:
            kv_cache = KVCache.cat(kv_cache)
            req.update_kv_cache(kv_cache)
            if req.input_ids_len() <= 0:
                kv_cache = kv_cache[:-1]
                req.update_kv_cache(kv_cache)

    def _cache_write_back(self, req: Request):
        # find again, if node already exists, skip
        self.trie_cache.insert(req.tokens()[:-1], req.kv_cache)

    def _sample(self, req: Request, logits: torch.Tensor) -> int:
        if req.temperature == 0:
            # greedy
            return int(logits.argmax(dim=-1).cpu())
        else:
            # sample (softmax with temperature)
            prob = torch.softmax(logits / req.temperature, dim=0)
            return int(torch.multinomial(prob, 1)[0])
            

    def _process_batch(self, reqs: Sequence[List], output):
        new_reqs = []
        for i, req in enumerate(reqs):
            next_token = self._sample(req, output.logits[i, -1])
            if next_token == 0:
                raise RuntimeError("Unexpected token generated")
            req.add_token(next_token)
            if self.use_cache:
                num_input_ids = req.input_ids_len() - 1
                kv_cache = KVCache(output.past_key_values)
                if req.has_cache():
                    new_kv = kv_cache[i, -num_input_ids:]
                    new_kv = KVCache.cat([req.kv_cache, new_kv])
                else:
                    new_kv = kv_cache[i, -num_input_ids:]
                req.update_kv_cache(new_kv)
            if req.finished():
                if self.use_cache:
                    self._cache_write_back(req)
                    req.update_kv_cache(None)
                req._event.set()
            else:
                new_reqs.append(req)
                # update kv_cache

        return new_reqs

    def _concat_kv_cache(self, reqs: Sequence[Request]) -> Optional[torch.Tensor]:
        # Concatenete kv_cache
        batch_size = len(reqs)
        start_t = time.time()
        kv_cache = [req.kv_cache.hf() if req.has_cache() else None for req in reqs]
        kv_len_list = [req.kv_len() for req in reqs]
        if not self.use_cache or sum(kv_len_list) == 0:
            max_kv_cache_len = 0
            kv_cache_tensors = None
            kv_len_list = [0] * batch_size
        else:
            max_kv_cache_len = max(kv_len_list)
            kv_example = [kv for kv in kv_cache if kv is not None][0]
            dtype = kv_example[0][0].dtype
            num_layers = len(kv_example)
            kv_shapes = [kv_example[0][i].shape for i in range(2)]

            kv_cache_tensors = []
            for layer in range(num_layers):
                layer_kv = []
                for k_or_v in range(2):
                    shape = kv_shapes[k_or_v]
                    num_heads, _, dim = shape
                    kv_tensor = torch.zeros(
                        (batch_size, num_heads, max_kv_cache_len, dim), device=self.device, dtype=dtype
                    )
                    for req_id in range(batch_size):
                        if kv_cache[req_id] is not None:
                            seqlen = kv_len_list[req_id]
                            kv_tensor[req_id, :, :seqlen].copy_(kv_cache[req_id][layer][k_or_v], non_blocking=True)
                    layer_kv.append(kv_tensor)
                kv_cache_tensors.append(layer_kv)
        self.build_kv_time += time.time() - start_t

        return kv_cache_tensors

    def _concat_input_ids(self, reqs: Sequence[Request]) -> torch.Tensor:
        batch_size = len(reqs)
        input_ids_list = [req.input_ids() for req in reqs]
        max_input_len = max([len(input_ids) for input_ids in input_ids_list])
        input_ids_tensor = torch.zeros((batch_size, max_input_len), device=self.device, dtype=torch.long)
        for i, input_ids in enumerate(input_ids_list):
            assert len(input_ids) > 0
            input_ids_tensor[i, -len(input_ids) :].copy_(torch.tensor(input_ids, device=self.device), non_blocking=True)
        return input_ids_tensor

    def _build_batch(self, reqs: Sequence[Request]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = len(reqs)

        # Concatenate KV Cache
        kv_cache_tensors = self._concat_kv_cache(reqs)
        max_kv_cache_len = 0 if kv_cache_tensors is None else kv_cache_tensors[0][0].shape[-2]

        # Concatenate input ids
        input_ids_tensor = self._concat_input_ids(reqs)
        max_input_len = input_ids_tensor.shape[1]

        # Construct mask
        attn_mask = torch.ones([batch_size, max_kv_cache_len + max_input_len], device=self.device, dtype=torch.long)
        for i, req in enumerate(reqs):
            # inputs left padding
            # kv right padding
            # kv_cache, 0, ..., 0, input_ids
            attn_mask[i, reqs[i].kv_len() : -req.input_ids_len()] = 0
        torch.cuda.synchronize()
        return input_ids_tensor, attn_mask, kv_cache_tensors

    @torch.inference_mode()
    def _run_model(self, *args, **kwargs):
        ret = self.model(*args, **kwargs)
        torch.cuda.synchronize()
        return ret

    @torch.inference_mode()
    def _main_loop(self):
        req_buf = []
        while not self.finished or len(req_buf) > 0:
            # get the next batch
            while not self.finished and len(req_buf) < self.max_batch_size:
                if len(req_buf) == 0:
                    req = self.queue.get()
                else:
                    try:
                        req = self.queue.get_nowait()
                    except queue.Empty:
                        break
                if req == "stop":
                    self.finished = True
                else:
                    if self.use_cache:
                        self._find_kv_cache(req)
                    req_buf.append(req)
            if len(req_buf) == 0:
                break
            print(f"[server step] batch size = {len(req_buf)}")
            input_ids, attn_mask, kv_cache_tensor = self._build_batch(req_buf)
            output = self._run_model(
                input_ids=input_ids, attention_mask=attn_mask, use_cache=self.use_cache, past_key_values=kv_cache_tensor
            )
            # sampling and cache write back
            req_buf = self._process_batch(req_buf, output)

    async def request(self, *args, **kwargs):
        if self.server_thread is None:
            raise RuntimeError("The server has not been launched yet.")
        req = Request(*args, **kwargs)
        self.queue.put(req)
        await asyncio.to_thread(req._event.wait)
        req.update_kv_cache(None)
        torch.cuda.empty_cache()
        return req

    def stop(self):
        self.queue.put("stop")
        self.server_thread.join()


async def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    args = parser.parse_args()
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model.half().to("cuda")
    # gpt2 is not supported
    server = PopukoServer(model=model, use_cache=args.cache)
    server.launch()
    input_ids = tokenizer(
        "This will be a very long prompt. So first of all, please tell me something about the university of Toronto. "
        "I'm mostly interested in its research reputation in the world, especially about Computer Science. Write something more about this."
    ).input_ids

    if args.check:
        ground_truth = model.generate(
            torch.tensor([input_ids], device="cuda", dtype=torch.long),  max_length=100
        )
        ground_truth = tokenizer.batch_decode(ground_truth)
        print(ground_truth)

    tasks = []
    torch.cuda.synchronize()
    start_t = time.time()
    for i in range(100):
        await asyncio.sleep(0.1)
        tasks.append(asyncio.create_task(server.request(input_ids, 10, eos_token_id=tokenizer.eos_token_id, temperature=0.7)))
    reqs = await asyncio.gather(*tasks)
    print("time elapsed:", time.time() - start_t, server.build_kv_time)
    server.stop()
    if args.check:
        for req in reqs:
            print(tokenizer.decode(req._generated_tokens))



if __name__ == "__main__":
    asyncio.run(_main())

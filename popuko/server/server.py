# A Simple LLM server supporting dynamic batching and
# exposing the control of kv-cache to users

from typing import Optional, List, Tuple

import threading
import queue
import time
import asyncio
import argparse

import transformers
import torch

from .cache import TrieCache

KVCache = Tuple[Tuple[torch.Tensor]]
# [Layer, kv, batchsize, num_heads, seqlen, dim]

class Request:
    def __init__(self, prompt : List[int], output_len=None, kv_cache=None):
        # Original request info
        # Should keep constant
        self.prompt = prompt
        self.kv_cache = kv_cache
        self.output_len = output_len

        # Runtime info
        self._generated_tokens = []
        self._event = threading.Event()
    
    def add_token(self, token: int):
        self._generated_tokens.append(token)
    
    def finished(self) -> bool:
        return len(self._generated_tokens) >= self.output_len
    
    def update_kv_cache(self, cache):
        self.kv_cache = cache
    
    def kv_len(self):
        if self.kv_cache is None:
            return 0
        else:
            return self.kv_cache[0][0].shape[1]

    def input_ids_len(self):
        return len(self.prompt) + len(self._generated_tokens) - self.kv_len()
    
    def input_ids(self):
        return (self.prompt + self._generated_tokens)[-self.input_ids_len():]


class PopukoServer:
    def __init__(self, model, max_batch_size:int=16, device='cuda', use_cache=True):
        self.model = model
        self.queue = queue.Queue()
        self.max_batch_size: int = max_batch_size
        self.finished = False

        self.trie_cache = TrieCache
        self.device = device
        self.use_cache = use_cache

        self.build_kv_time = 0
        self.server_thread = None

    def launch(self):
        self.server_thread = threading.Thread(target=self._main_loop)
        self.server_thread.start()
    
    def find_kv_cache(self, req: Request) -> KVCache:
        # look up kv cache in the trie
        return req.kv_cache

    def process_batch(self, reqs, output):
        new_reqs = []
        for i, req in enumerate(reqs):
            next_token = int(output.logits[i, req.input_ids_len() - 1].argmax(dim=-1).cpu())
            req.add_token(next_token)
            if req.finished():
                req.update_kv_cache(None)
                torch.cuda.empty_cache()
                req._event.set()
            else:
                new_reqs.append(req)
                # update kv_cache
                if self.use_cache:
                    seq_len = req.input_ids_len() - 1
                    new_kv = [[kv[i, :, :seq_len, ].clone() for kv in layer] for layer in output.past_key_values]
                    req.update_kv_cache(new_kv)
        # del output.past_key_values
        torch.cuda.empty_cache()
        return new_reqs

    def build_batch(self, reqs):
        batch_size = len(reqs)
        kv_cache = [req.kv_cache for req in reqs]
        kv_len_list = [0 if req_cache is None else req_cache[0][0].shape[1] for req_cache in kv_cache]

        # Concatenete kv_cache
        start_t = time.time()
        if not self.use_cache or sum(kv_len_list) == 0:
            max_kv_cache_len = 0
            kv_cache_tensor = None
            kv_len_list = [0] * batch_size
        else:
            max_kv_cache_len = max(kv_len_list)
            kv_example = [kv for kv in kv_cache if kv is not None][0]
            dtype = kv_example[0][0].dtype
            num_layers = len(kv_example)
            kv_shapes = [kv_example[0][i].shape for i in range(2)]

            kv_cache_tensor = []
            for layer in range(num_layers):
                layer_kv = []
                for k_or_v in range(2):
                    shape = kv_shapes[k_or_v]
                    num_heads, _, dim = shape
                    kv_tensor = torch.empty((batch_size, num_heads, max_kv_cache_len, dim), device=self.device, dtype=dtype)
                    for req_id in range(batch_size):
                        if kv_cache[req_id] is not None:
                            seqlen = kv_len_list[req_id]
                            kv_tensor[req_id, :, :seqlen].copy_(kv_cache[req_id][layer][k_or_v], non_blocking=True)
                    layer_kv.append(kv_tensor)
                kv_cache_tensor.append(layer_kv)
        self.build_kv_time += time.time() - start_t 

        # Concatenate input ids
        input_ids_list = [(req.prompt + req._generated_tokens)[kv_len:] for req, kv_len in zip(reqs, kv_len_list)]
        max_input_len = max([len(input_ids) for input_ids in input_ids_list])

        input_ids_tensor = torch.zeros((batch_size, max_input_len), device=self.device, dtype=torch.long)
        for i, input_ids in enumerate(input_ids_list):
            input_ids_tensor[i, :len(input_ids)].copy_(torch.tensor(input_ids, device=self.device), non_blocking=True)

        # Construct mask
        attn_mask = torch.ones([batch_size, max_kv_cache_len + max_input_len], device=self.device, dtype=torch.long)
        for i, input_ids in enumerate(input_ids_list):
            attn_mask[i, max_kv_cache_len + len(input_ids):] = 0
            attn_mask[i, kv_len_list[i] : max_kv_cache_len] = 0

        torch.cuda.synchronize()
        return input_ids_tensor, attn_mask, kv_cache_tensor
    
    # @torch.inference_mode()
    def run_model(self, *args, **kwargs):
        ret = self.model(*args, **kwargs)
        torch.cuda.synchronize()
        return ret

    @torch.inference_mode()
    def _main_loop(self):
        print('The server is launched')
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
                if req == 'stop':
                    print('server exiting')
                    self.finished = True
                else:
                    req_buf.append(req)
                    # print('recv', id(req))
            if len(req_buf) == 0:
                print('exiting')
                break
            input_ids, attn_mask, kv_cache_tensor = self.build_batch(req_buf)
            output = self.run_model(input_ids=input_ids, 
                                    attention_mask=attn_mask, 
                                    use_cache=self.use_cache, 
                                    past_key_values=kv_cache_tensor)
            torch.cuda.empty_cache()
            req_buf = self.process_batch(req_buf, output)
    
    async def request(self, *args, **kwargs):
        req = Request(*args, **kwargs)
        self.queue.put(req)
        await asyncio.to_thread(req._event.wait)
        req.update_kv_cache(None)
        torch.cuda.empty_cache() 
        return req
    
    def stop(self):
        self.queue.put('stop')
        self.server_thread.join()


async def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action='store_true')
    parser.add_argument("--check", action='store_true')
    args = parser.parse_args()
    model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-large')
    model.half().to('cuda')
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-large')
    server = PopukoServer(model=model, use_cache=args.cache)
    server.launch()
    input_ids = tokenizer('Tell me something about the university of Toronto').input_ids
    tasks = []
    start_t = time.time()
    for i in range(100):
        await asyncio.sleep(0.1)
        tasks.append(server.request(input_ids, 100))
    reqs = await asyncio.gather(*tasks)
    server.stop()
    print(time.time() - start_t, server.build_kv_time)
    if args.check:
        for req in reqs:
            print(tokenizer.decode(req._generated_tokens))

if __name__ == '__main__':
    asyncio.run(_main())
# A Simple LLM server supporting dynamic batching and
# exposing the control of kv-cache to users

import threading
import queue
import time

import transformers
import torch

class Request:
    def __init__(self, prompt, kv_cache=None, output_len=None):
        self.ready = False
        self.prompt = prompt
        self.kv_cache = kv_cache
        self.output_len = output_len
    
    def find_kv_cache(self):
        # look up kv cache in the trie
        pass

class PopukoServer:
    def __init__(self, model, max_batch_size=16):
        self.model = model
        self.queue = queue.Queue()
        self.max_batch_size = max_batch_size

    def launch(self):
        self.server_thread = threading.Thread(target=self._main_loop)
        self.server_thread.start()
        print('The server is launched')
    
    def stop(self):
        self.server_thread.join()
    
    def _main_loop(self):
        req_buf = []
        while True:
            # get the next batch
            while len(req_buf) < self.max_batch_size:
                try:
                    req = self.queue.get_nowait()
                    req_buf.append(req)
                except queue.Empty:
                    break
            if len(req_buf) == 0:
                time.sleep(0.1)
                continue
            print("size of req buf:", len(req_buf))
            req_buf.clear()
    
    def add_request(self, req):
        self.queue.put(req)
        # print("put a request")
    

if __name__ == '__main__':
    server = PopukoServer(None)
    server.launch()
    for i in range(100):
        server.add_request('DOG')
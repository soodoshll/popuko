# A simple LLM Server

This module is a simple LLM server supports

 * Dynamic batching (continuous batching)
 * `asyncio` interface
 * KV-Cache (`past_key_values`)
 * Cross-query kvcache based on a trie structure

It has been tested on the following models:

 * OPT (125m, 1.3b, 6.7b)
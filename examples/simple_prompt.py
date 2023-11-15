import popuko
from popuko import query, generate

import asyncio

if __name__ == "__main__":
    popuko.init_hf_llm('facebook/opt-1.3b')

    @query(' ')
    def example():
        "The University of Toronto is located in"
        generate(max_new_tokens=10, temperature=0.9)
        "The number of students is"
        generate(max_new_tokens=5, temperature=0.9)

    print(asyncio.run(example()))

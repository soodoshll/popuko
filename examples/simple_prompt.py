import popuko
from popuko import query, generate

import asyncio

if __name__ == "__main__":
    popuko.init_llm('facebook/opt-125m')

    @query(' ')
    def example():
        "The University of Toronto is located in"
        A = generate(max_new_tokens=10, temperature=0.9)
        A
        "The number of students is"
        B = generate(max_new_tokens=5, temperature=0.9)
        B

    out = asyncio.run(example().wait())
    print(out)
import popuko
from popuko import query, generate

import asyncio

if __name__ == "__main__":
    popuko.init_llm('facebook/opt-125m')

    @query(' ')
    def uoft_intro():
        "The University of Toronto is located in"
        generate(max_new_tokens=10, temperature=0.9)
        "The number of students is"
        generate(max_new_tokens=5, temperature=0.9)
    
    @query(' ')
    def more_thing(intro):
        "Here is brief introduction of UofT:"
        intro
        ". In conclusion,"
        generate(max_new_tokens=20)

    async def main():
        more_future = more_thing(uoft_intro())
        return await more_future.wait()

    out = popuko.get_event_loop().run_until_complete(main())
    print(out)



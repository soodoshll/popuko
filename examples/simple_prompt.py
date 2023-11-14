import popuko
from popuko import query, generate

popuko.init_hf_llm('facebook/opt-6.7b')

if __name__ == "__main__":
    @query
    def example():
        "The University of Toronto is located in"
        A = generate()
        "The number of students is"
        B = generate()

    ret = example()
    print(ret)

import popuko
from popuko import query, generate

popuko.init_hf_llm('facebook/opt-6.7b')

if __name__ == "__main__":
    @query
    def example():
        "The University of Toronto is located in"
        generate()
        "The number of students is"
        generate()

    print(example())

import tiktoken
enc = tiktoken.get_encoding("o200k_base")
#assert enc.decode(enc.encode("hello world")) == "hello world"
r = enc.encode("<|begin_of_text|>The cat sat on the mat.")

print(r)


# To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("meta-llama/Meta-Llama-3-8B")
# print(enc)
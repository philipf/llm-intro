from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-bnb-4bit")

prompt = "The cat sat on the"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print("Inputs", inputs)

outputs = model.generate(**inputs, max_new_tokens=2, temperature=0.01)

print("Outputs", outputs)

decoded_output = tokenizer.decode(outputs[0])
print("Decoded output", decoded_output)
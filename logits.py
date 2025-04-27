import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-bnb-4bit")

# Prompt and inputs
prompt = "The cat sat on the"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Get logits without gradients
with torch.no_grad():
    outputs = model(**inputs)


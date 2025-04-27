import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-bnb-4bit").to("cuda")

# Prompt and inputs
prompt = "The cat sat in the"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Get logits without gradients
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
last_token_logits = logits[0, -1]

# GUI Plotting setup
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

# Initial temperature
init_temp = 1.0

def update_plot(temp):
    ax.clear()

    # Apply temperature
    scaled_logits = last_token_logits / temp
    probs = torch.softmax(scaled_logits, dim=-1)

    # Get top tokens
    top_probs, top_indices = torch.topk(probs, k=10)
    top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())

    # Plot
    ax.bar(top_tokens, top_probs.cpu().numpy())
    ax.set_title(f"Top Predicted Next Tokens (Temp = {temp:.2f})")
    ax.set_xlabel("Token")
    ax.set_ylabel("Probability")
    fig.canvas.draw_idle()

# Initial plot
update_plot(init_temp)

# Add temperature slider
ax_temp = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_temp, 'Temperature', 0.1, 2.0, valinit=init_temp, valstep=0.05)

# Attach update function
slider.on_changed(update_plot)

plt.show()

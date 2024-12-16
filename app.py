from flask import Flask, render_template, request, session, redirect, url_for
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import os
from datetime import timedelta

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Replace with a secure secret key in production
app.permanent_session_lifetime = timedelta(days=7)  # Session lifetime

# Configuration - Replace with your Hugging Face adapter path
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"  # Base model from Unsloth
ADAPTER_MODEL = "Rymadellai/beauty_of_tunisia"  # Your PEFT adapter path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
print(f"Loading base model '{BASE_MODEL}' with adapter '{ADAPTER_MODEL}'...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load the PEFT adapter model
model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_MODEL,
    device_map="auto",   # Automatically map model to GPU if available
    load_in_4bit=True,   # Load model in 4-bit quantization
)
print("Model loaded successfully!")

# Define the bipolar-specific prompt template
bipolar_prompt = """You are an expert in Tunisia tourism. Your task is to help tourists gain information about Tunisian places, prices, and everything else they need to know.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if 'history' not in session:
        session['history'] = []

    if request.method == "POST":
        instruction = request.form.get("instruction", "").strip()
        user_input = request.form.get("user_input", "").strip()

        if instruction and user_input:
            # Prepare the prompt
            prompt = bipolar_prompt.format(instruction, user_input, "")

            # Tokenize the input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Ensure prompt fits within the model's context window
            ).to(DEVICE)

            try:
                # Generate the response
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id  # Ensure EOS is used for padding
                )

                # Decode the generated text
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text.replace(prompt, "").strip()

                # Update session history
                session['history'].append({
                    'instruction': instruction,
                    'user_input': user_input,
                    'response': response
                })
                session.modified = True  # Mark the session as modified to ensure it's saved
            except Exception as e:
                response = f"An error occurred during generation: {str(e)}"
        else:
            response = "Please provide both instruction and input."

    return render_template("index.html", history=session.get('history', []))

@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop('history', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)

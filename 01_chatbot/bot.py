import os
import json
import torch
import streamlit as st
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_PATH = "fine_tuned_model"

with open("Hasnain.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def format_example(example):
    return {
        "text": f"Instruction: {example['instruction']}\nResponse: {example['output']}"
    }

dataset = Dataset.from_list(data)
dataset = dataset.map(format_example)

if not os.path.exists(MODEL_PATH):
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example

    tokenized_dataset = tokenized_dataset.map(add_labels)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("✅ Model trained and saved!")

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ========== STREAMLIT UI WITH HASNAIN BRANDING ==========
st.set_page_config(
    page_title="Hasnain AI - Your Intelligent Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    /* Chat bubbles */
    .user-bubble {
        background-color: #e0f2fe;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
    }
    .ai-bubble {
        background-color: #f3e8ff;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
    }
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Display header with name
st.markdown('<div class="main-title">🧠 Hasnain AI 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent conversational companion</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history with bubble styling
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div style="text-align: right;"><span class="user-bubble">👤 You: {msg["content"]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align: left;"><span class="ai-bubble">🤖 Hasnain AI: {msg["content"]}</span></div>', unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("✍️ Type your message here...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div style="text-align: right;"><span class="user-bubble">👤 You: {prompt}</span></div>', unsafe_allow_html=True)

    # Exact match from JSON
    response = None
    for item in data:
        if prompt.lower().strip() == item["instruction"].lower().strip():
            response = item["output"]
            break

    if response is None:
        input_text = f"Instruction: {prompt}\nResponse:"
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(input_text):].strip()
        if not response:
            response = "I'm not sure how to respond to that. Can you rephrase?"

    # Show AI response
    st.markdown(f'<div style="text-align: left;"><span class="ai-bubble">🤖 Hasnain AI: {response}</span></div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with your name
st.markdown(f"""
    <div class="footer">
        Developed with ❤️ by <strong>Chaudhary Hasnain</strong> | © 2026
    </div>
""", unsafe_allow_html=True)
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
import os

# Hugging Face access token (stored in HF Spaces secrets)
hf_token = os.getenv("HF_TOKEN")

adapter_path = "imnim/multi-label-email-classifier"

# Load PEFT config
peft_config = PeftConfig.from_pretrained(adapter_path, token=hf_token)

# Load base model (fallback to float32 if bfloat16 fails)
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )
except:
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float32,
        device_map="auto",
        token=hf_token
    )

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, token=hf_token)

model = PeftModel.from_pretrained(base_model, adapter_path, token=hf_token)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define classification function
def classify_email(subject, body):
    prompt = f"""### Subject:\n{subject}\n\n### Body:\n{body}\n\n### Labels:"""
    result = pipe(prompt, max_new_tokens=50, do_sample=True, top_k=50, top_p=0.95)
    full_text = result[0]["generated_text"]
    label_section = full_text.split("### Labels:")[1].strip()
    return label_section

# Gradio UI
demo = gr.Interface(
    fn=classify_email,
    inputs=["text", "text"],
    outputs="text",
    title="Multi-label Email Classifier",
    description="Enter subject and body to get label prediction"
)

demo.launch()

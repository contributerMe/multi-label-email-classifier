from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapter_path = "imnim/multi-label-email-classifier"

try:
    # Load PEFT config to get base model path
    peft_config = PeftConfig.from_pretrained(adapter_path, use_auth_token=True)
    
    # Load base model and tokenizer with HF auth token
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        use_auth_token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        use_auth_token=True
    )

    # Load adapter with HF auth token
    model = PeftModel.from_pretrained(
        base_model, adapter_path,
        device_map={"": "cpu"},
        use_auth_token=True
    )

    # Setup text-generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

except Exception as e:
    raise RuntimeError(f"❌ Failed to load model + adapter: {str(e)}")

# Request schema
class EmailInput(BaseModel):
    subject: str
    body: str

# POST /classify endpoint
@app.post("/classify")
async def classify_email(data: EmailInput):
    prompt = f"""### Subject:\n{data.subject}\n\n### Body:\n{data.body}\n\n### Labels:"""
    try:
        result = pipe(prompt, max_new_tokens=50, do_sample=True, top_k=50, top_p=0.95)
        full_text = result[0]["generated_text"]
        label_section = full_text.split("### Labels:")[1].strip()
        return {"label": label_section}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
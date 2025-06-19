import sys
import os

# Add the root project directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from base_model_data_creation.run_base_model import get_model_output
from utils import get_quantized_model
import torch
import traceback

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATOR_MODEL_ID = "CohereLabs/c4ai-command-r-v01"

# Jury model names
jury1 = "olmo13b"
jury2 = "llama13b"
jury3 = "stable13b"

# Global model store
generator = {}

@app.on_event("startup")
def load_model():
    global tok_mod
    try:
        print("🔧 Loading generator model...")
        gen_mod = get_quantized_model(GENERATOR_MODEL_ID, device)
        gen_tok = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)
        gen_mod.eval()
        generator["tokenizer"] = gen_tok
        generator["model"] = gen_mod
    except Exception as e:
        return process_error(e)

class EvalRequest(BaseModel):
    question: str

@app.post("/generate")
def generate(eval_req: EvalRequest):
    try:
        question = eval_req.question.strip()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_tok, gen_mod = generator["tokenizer"], generator["model"]
        output_text = get_model_output(gen_mod, gen_tok, question, device)
        return { "question": question, "generated_answer": output_text }
    except Exception as e:
        return process_error(e)

def process_error(e):
    traceback_str = traceback.format_exc()
    print("🔥 Internal error:\n", traceback_str)  # Print in Colab logs
    return JSONResponse(status_code=500, content={"error": str(e)})
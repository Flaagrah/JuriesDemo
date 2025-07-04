import sys
import os

# Add the root project directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from base_model_data_creation.run_base_model import get_model_output
from utils import get_epsilon_dict, shuffle_jury_data, calculate_metrics_for_response, get_quantized_model, filter_and_shuffle_jury_data, FILTERED_SUFFIX
from jury_finetuning.judge_response import call_jury_on_single_prompt
import torch
import traceback

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Jury model names
jury1 = "olmo13b"
jury2 = "llama13b"
jury3 = "stable13b"

# Three jury models stored locally
JURY_MODEL_PATHS = {
    jury1: "qlora_olmo13b_finetuned/checkpoint-313",
    jury2: "qlora_open_llama_13b_finetuned/checkpoint-313",
    jury3: "qlora_stable13b_finetuned/checkpoint-313",
}

# Three jury models' epsilon values
JURY_EPSILONS = {
    jury1: {},
    jury2: {},
    jury3: {}
}

# Global model store
models = {}

@app.on_event("startup")
def load_models():
    try:
        print("🔧 Loading local jury models...")
        for name, path in JURY_MODEL_PATHS.items():
            tok = AutoTokenizer.from_pretrained(path)
            # Load the model with 4-bit quantization.
            mod = get_quantized_model(path, device)
            mod.eval()
            models[name] = (tok, mod)

        print("✅ All models loaded.")

        print("Shuffling calibration data...")
        seed = 9
        jury_dict = {
            "olmo13b": {
                "calib_name": "qlora_olmo13b_calib",
                "test_name": "qlora_olmo13b_test"
            },
            "llama13b": {
                "calib_name": "qlora_llama13b_calib",
                "test_name": "qlora_llama13b_test"
            },
            "stable13b": {
                "calib_name": "qlora_stable13b_calib",
                "test_name": "qlora_stable13b_test"
            }
        }
        filter_and_shuffle_jury_data(jury_dict, seed=seed, filter_disagreements=True)
        # shuffle_jury_data("qlora_olmo13b_calib", "qlora_olmo13b_test", seed)
        # shuffle_jury_data("qlora_llama13b_calib", "qlora_llama13b_test", seed)
        # shuffle_jury_data("qlora_stable13b_calib", "qlora_stable13b_test", seed)

        JURY_EPSILONS[jury1] = get_epsilon_dict("qlora_"+jury1+"_calib"+FILTERED_SUFFIX+"_shuffled")
        JURY_EPSILONS[jury2] = get_epsilon_dict("qlora_"+jury2+"_calib"+FILTERED_SUFFIX+"_shuffled")
        JURY_EPSILONS[jury3] = get_epsilon_dict("qlora_"+jury3+"_calib"+FILTERED_SUFFIX+"_shuffled")
        print("JURY_EPSILONS:", JURY_EPSILONS)
        print("✅ Calibration data shuffled and epsilon values loaded.")
    except Exception as e:
        return process_error(e)

class EvalRequest(BaseModel):
    question: str
    answer: str

@app.post("/evaluate")
def evaluate(eval_req: EvalRequest):
    try:
        question = eval_req.question.strip()
        answer = eval_req.answer.strip()

        jury_to_metrics = {}
        for jury_name, (jury_tok, jury_mod) in models.items():
            logits = call_jury_on_single_prompt(
                jury_mod, jury_tok, question, answer
            )
            jury_to_metrics[jury_name] = {
                "logits": logits,
                "epsilon_to_s": JURY_EPSILONS[jury_name],
            }

        jury_agg_judgements = calculate_metrics_for_response(jury_to_metrics)
        jury_agg_judgements["generated_answer"] = answer
        jury_agg_judgements["question"] = question
        # Step 3: Return results
        return jury_agg_judgements
    except Exception as e:
        return process_error(e)

def process_error(e):
    traceback_str = traceback.format_exc()
    print("🔥 Internal error:\n", traceback_str)  # Print in Colab logs
    return JSONResponse(status_code=500, content={"error": str(e)})
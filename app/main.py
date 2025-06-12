import sys
import os

# Add the root project directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from base_model_data_creation.run_base_model import get_model_output
from utils import get_epsilon_dict, shuffle_jury_data, calculate_metrics_for_response
from jury_finetuning.judge_response import call_jury_on_single_prompt
import torch

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One generator from Hugging Face
GENERATOR_MODEL_ID = "CohereLabs/c4ai-command-r7b-12-2024"

# Jury model names
jury1 = "olmo13b"
jury2 = "llama13b"
jury3 = "stable13b"

# Three jury models stored locally
JURY_MODEL_PATHS = {
    jury1: "qlora_olmo13b_finetuned/checkpoint-625",
    jury2: "qlora_open_llama_13b_finetuned/checkpoint-625",
    jury3: "qlora_stable13b_finetuned/checkpoint-625",
}

# Three jury models' epsilon values
JURY_EPSILONS = {
    jury1: {},
    jury2: {},
    jury3: {}
}

# Global model store
models = {}

#@app.on_event("startup")
def load_models():
    print("🔧 Loading generator model...")
    gen_tok = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)
    gen_mod = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    ).to(device)
    gen_mod.eval()
    models["generator"] = (gen_tok, gen_mod)

    print("🔧 Loading local jury models...")
    for name, path in JURY_MODEL_PATHS.items():
        tok = AutoTokenizer.from_pretrained(path)
        # Set up 4-bit quantization configuration via bitsandbytes.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat 4 (NF4) quantization
            bnb_4bit_use_double_quant=True,       # Use double quantization for improved accuracy
            bnb_4bit_compute_dtype=torch.bfloat16 # Use BF16 for compute
        )

        # # Load the model with 4-bit quantization.
        mod = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto"
        ).to(device)
        mod.eval()
        models[name] = (tok, mod)

    print("✅ All models loaded.")

    print("Shuffling calibration data...")
    seed = 9
    shuffle_jury_data("qlora_olmo13b_calib", "qlora_olmo13b_test", seed)
    shuffle_jury_data("qlora_llama13b_calib", "qlora_llama13b_test", seed)
    shuffle_jury_data("qlora_stable13b_calib", "qlora_stable13b_test", seed)

    JURY_EPSILONS[jury1] = get_epsilon_dict("qlora_"+jury1+"_calib_shuffled")
    JURY_EPSILONS[jury2] = get_epsilon_dict("qlora_"+jury2+"_calib_shuffled")
    JURY_EPSILONS[jury3] = get_epsilon_dict("qlora_"+jury3+"_calib_shuffled")

    return JURY_EPSILONS

#@app.post("/evaluate")
def evaluate(question: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 1: Generate answer
    gen_tok, gen_mod = models["generator"]
    output_text = get_model_output(gen_mod, gen_tok, question, device)

    jury_to_metrics = {}
    # Step 2: Evaluate with all 3 juries
    for jury_name, (jury_tok, jury_mod) in models.items():
        if jury_name == "generator":
            continue
        # Call the jury model on the generated answer
        logits = call_jury_on_single_prompt(
            jury_mod, jury_tok, question, output_text
        )
        jury_to_metrics[jury_name] = {
            "logits": logits,
            "epsilon_to_s": JURY_EPSILONS[jury_name],
        }

    jury_agg_judgements = calculate_metrics_for_response(jury_to_metrics)
    jury_agg_judgements["generated_answer"] = output_text
    jury_agg_judgements["question"] = question
    # Step 3: Return results
    return jury_agg_judgements

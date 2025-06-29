import pandas as pd
from stats import generate_stats, generate_stats2, aggregate_stats
from utils import get_epsilon_dict, get_test_s_vals, filter_and_shuffle_jury_data, FILTERED_SUFFIX

agg_data = []
for i in range(0, 10): 

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
    filter_and_shuffle_jury_data(jury_dict, seed=i, filter_disagreements=True)

    olmo13b_alphas = get_epsilon_dict("qlora_olmo13b_calib"+FILTERED_SUFFIX+"_shuffled")
    llama13b_alphas = get_epsilon_dict("qlora_llama13b_calib"+FILTERED_SUFFIX+"_shuffled")
    stable13b_alphas = get_epsilon_dict("qlora_stable13b_calib"+FILTERED_SUFFIX+"_shuffled")

    olmo13b_test = get_test_s_vals("qlora_olmo13b_test"+FILTERED_SUFFIX+"_shuffled")
    llama13b_test = get_test_s_vals("qlora_llama13b_test"+FILTERED_SUFFIX+"_shuffled")
    stable13b_test = get_test_s_vals("qlora_stable13b_test"+FILTERED_SUFFIX+"_shuffled")

    results = generate_stats(llama13b_test, olmo13b_test, stable13b_test, llama13b_alphas, olmo13b_alphas, stable13b_alphas)
    print("-----------------------------------------------------------")
    #generate_stats2(llama13b_test, olmo13b_test, stable13b_test, llama13b_alphas, olmo13b_alphas, stable13b_alphas)
    print("-----------------------------------------------------------")
    agg_data.append(results)
print("-----------------------------------------------------------")
print("Aggregated Results:")
print(aggregate_stats(agg_data))

'''
Seed 9, with filtering
Majority Poll: 107 86 0.5544041450777202
Calibrated Confidence Poll: 115 78 0.5958549222797928
Calibrated Mul Confidence Poll: 112 81 0.5803108808290155
Calibrated Max Poll: 119 74 0.616580310880829
Max Poll (Uncalibrated): 115 78 0.5958549222797928
Veto Poll: 116 77 0.6010362694300518
Majorities: 111 82
True Majority: 51 60
False Majority: 56 26
Start Interval: 0.5
End Interval: 1
Llama correct: 102 91 0.5284974093264249
Olmo correct: 84 109 0.43523316062176165
Stable correct: 107 86 0.5544041450777202
Base Model Correct: 77 116
Max Logits: [82, 70, 41]
Max Confidence: [88, 50, 55]
Jury Correct on disagreements: [102, 84, 107]
Jury Correct on disagreements (normalized): [0.5284974093264249, 0.43523316062176165, 0.5544041450777202]
Majority Vote: 107 86 0.5544041450777202
Calibrated Confidence Poll: 115 78 0.5958549222797928
Calibrated Multiplicative Poll: 112 81 0.5803108808290155
Calibrated Max Poll: 119 74 0.616580310880829
Max Poll (Uncalibrated): 115 78 0.5958549222797928
Veto Poll: 116 77 0.6010362694300518
'''
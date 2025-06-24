from print_stats import print_stats, print_stats2
from utils import get_epsilon_dict, get_test_s_vals, filter_and_shuffle_jury_data, FILTERED_SUFFIX

dist_list = []
for i in range(9, 10):  
    # shuffle_jury_data("qlora_olmo13b_calib", "qlora_olmo13b_test", seed = i)
    # shuffle_jury_data("qlora_llama13b_calib", "qlora_llama13b_test", seed = i)
    # shuffle_jury_data("qlora_stable13b_calib", "qlora_stable13b_test", seed = i)

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

    print(olmo13b_alphas)
    print(llama13b_alphas)
    print(stable13b_alphas)

    olmo13b_test = get_test_s_vals("qlora_olmo13b_test"+FILTERED_SUFFIX+"_shuffled")
    llama13b_test = get_test_s_vals("qlora_llama13b_test"+FILTERED_SUFFIX+"_shuffled")
    stable13b_test = get_test_s_vals("qlora_stable13b_test"+FILTERED_SUFFIX+"_shuffled")

    # olmo13b_alphas = get_epsilon_dict("qlora_olmo13b_calib_shuffled")
    # llama13b_alphas = get_epsilon_dict("qlora_llama13b_calib_shuffled")
    # stable13b_alphas = get_epsilon_dict("qlora_stable13b_calib_shuffled")

    # olmo13b_test = get_test_s_vals("qlora_olmo13b_test_shuffled")
    # llama13b_test = get_test_s_vals("qlora_llama13b_test_shuffled")
    # stable13b_test = get_test_s_vals("qlora_stable13b_test_shuffled")

    results = print_stats(llama13b_test, olmo13b_test, stable13b_test, llama13b_alphas, olmo13b_alphas, stable13b_alphas)
    print_stats2(llama13b_test, olmo13b_test, stable13b_test, llama13b_alphas, olmo13b_alphas, stable13b_alphas)
    print("-----------------------------------------------------------")
    #dist_list.append(results)
# print(dist_list)
# columns = ["Majority Poll", "Calibrated Confidence Poll", "Calibrated Mul Confidence Poll", "Calibrated Max Poll", "Max Poll (Uncalibrated)"]
# dist_df = pd.DataFrame(dist_list, columns=columns)
# print(dist_df)
# dist_df.to_csv('results.csv')
'''
Seed 9
Accurate vs Inaccurate judgements
Majority Poll: 79 43 0.6475409836065574
Calibrated Confidence Poll: 79 43 0.6475409836065574
Calibrated Mul Confidence Poll: 79 43 0.6475409836065574
Calibrated Max Poll: 89 33 0.7295081967213115
Max Poll (Uncalibrated): 81 41 0.6639344262295082
Veto Poll: 70 52 0.5737704918032787
Majorities: 67 55
True Majority: 38 29
False Majority: 41 14
Start Interval: 0.5
End Interval: 1
Llama correct: 886 114 0.886
Olmo correct: 910 90 0.91
Stable correct: 907 93 0.907
Base Model Correct: 365 635
Max Logits: [46, 37, 39]
Max Confidence: [31, 45, 46]
Jury Correct on disagreements: [52, 76, 73]
Jury Correct on disagreements (normalized): [0.4262295081967213, 0.6229508196721312, 0.5983606557377049]
'''
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
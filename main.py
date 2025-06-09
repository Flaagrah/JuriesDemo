from print_stats import print_stats, print_stats2
from utils import get_epsilon_dict, shuffle_jury_data, get_test_s_vals

dist_list = []
for i in range(9, 10):  
    shuffle_jury_data("qlora_olmo13b_calib", "qlora_olmo13b_test", seed = i)
    shuffle_jury_data("qlora_llama13b_calib", "qlora_llama13b_test", seed = i)
    shuffle_jury_data("qlora_stable13b_calib", "qlora_stable13b_test", seed = i)

    olmo13b_alphas = get_epsilon_dict("qlora_olmo13b_calib_shuffled")
    llama13b_alphas = get_epsilon_dict("qlora_llama13b_calib_shuffled")
    stable13b_alphas = get_epsilon_dict("qlora_stable13b_calib_shuffled")

    olmo13b_test = get_test_s_vals("qlora_olmo13b_test_shuffled")
    llama13b_test = get_test_s_vals("qlora_llama13b_test_shuffled")
    stable13b_test = get_test_s_vals("qlora_stable13b_test_shuffled")

    results = print_stats(llama13b_test, olmo13b_test, stable13b_test, llama13b_alphas, olmo13b_alphas, stable13b_alphas)
    print_stats2(llama13b_test, olmo13b_test, stable13b_test, llama13b_alphas, olmo13b_alphas, stable13b_alphas)
    print("-----------------------------------------------------------")
    dist_list.append(results)
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


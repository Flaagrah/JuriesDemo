
from find_epsilon_from_s import find_epsilon_from_s
from pandas import DataFrame

def print_stats(llama_test_vals: DataFrame, olmo_test_vals: DataFrame, stable_test_vals: DataFrame, llama_dict: dict, olmo_dict: dict, stable_dict: dict):
    
    total_disagreements = 0
    
    # majority poll vs calibrated poll
    accurate_judgements = 0
    cal_acc_judgements = 0
    cal_mul_acc_judgements = 0
    max_cal_acc_judgements = 0
    max_acc_judgements = 0
    veto_acc_judgements = 0

    true_majority = 0
    false_majority = 0
    true_majority_true = 0
    true_majority_false = 0
    false_majority_true = 0
    false_majority_false = 0

    all_vals = []

    llama_pd_data = []
    olmo_pd_data = []
    stable_pd_data = []

    interval_start = 0.5
    interval_end = 1

    jury_correct = [0, 0, 0]

    base_model_correct = 0

    for i in range(len(llama_test_vals)):
        
        ll_info = llama_test_vals.loc[i]
        ol_info = olmo_test_vals.loc[i]
        st_info = stable_test_vals.loc[i]

        if ll_info['is_base_model_correct']:
            base_model_correct += 1
        
        llama_confidence = 1 - find_epsilon_from_s(ll_info['s_val'], llama_dict)
        olmo_confidence = 1 - find_epsilon_from_s(ol_info['s_val'], olmo_dict)
        stable_confidence = 1 - find_epsilon_from_s(st_info['s_val'], stable_dict)

        info_list = [{'data': ll_info, 'confidence': llama_confidence}, 
                     {'data': ol_info, 'confidence': olmo_confidence}, 
                     {'data': st_info, 'confidence': stable_confidence}]

        # Loop through info_list
        for index, information in enumerate(info_list):
            if information['confidence'] >= interval_start and information['confidence'] <= interval_end:
                if information['data']['is_base_model_correct'] == information['data']['is_jury_approving']:
                    jury_correct[index] += 1

        llama_pd_data.append([ll_info['s_val'], llama_confidence, ll_info['is_base_model_correct'] == ll_info['is_jury_approving'], ll_info['is_jury_approving']])
        olmo_pd_data.append([ol_info['s_val'], olmo_confidence, ol_info['is_base_model_correct'] == ol_info['is_jury_approving'], ol_info['is_jury_approving']])
        stable_pd_data.append([st_info['s_val'], stable_confidence, st_info['is_base_model_correct'] == st_info['is_jury_approving'], st_info['is_jury_approving']])

        true_counts = 0
        false_counts = 0
        correct_score = 0
        incorrect_score = 0
        correct_mul_score = 1
        incorrect_mul_score = 1

        for index, information in enumerate(info_list):
            if information['data']['is_jury_approving']:
                true_counts += 1
                correct_score += information['confidence'] - 0.5
                correct_mul_score *= (1 - information['confidence'])
            else:
                false_counts += 1
                incorrect_score += information['confidence'] - 0.5
                incorrect_mul_score *= (1 - information['confidence'])

        # Continue if they all agree
        if true_counts == 0 or false_counts == 0:
            continue

        total_disagreements += 1

        # Count veto judgements
        if false_counts > 0:
            if not ll_info['is_base_model_correct']:
                veto_acc_judgements += 1

        # Count majorities
        if true_counts > false_counts:
            true_majority += 1
            if ll_info['is_base_model_correct']:
                true_majority_true += 1
            else:
                true_majority_false += 1
        else: 
            false_majority += 1
            if ll_info['is_base_model_correct']:
                false_majority_true += 1
            else:
                false_majority_false += 1
        sorted = [round(llama_confidence - 0.5, 2), round(olmo_confidence - 0.5, 2), round(stable_confidence - 0.5, 2)]
        all_vals.append(sorted)    
        #print(sorted)
        # For each entry in sorted, get the index of the max entry
        max_index = sorted.index(max(sorted))
        # Max Confidence
        if info_list[max_index]['data']['is_base_model_correct'] == info_list[max_index]['data']['is_jury_approving']:
            max_cal_acc_judgements += 1

        sorted = [round(ll_info['s_val'], 4), round(ol_info['s_val'], 4), round(st_info['s_val'], 4)]
        #print(sorted)
        # For each entry in sorted, get the index of the max entry
        max_index = sorted.index(max(sorted))
        # Max Confidence
        if info_list[max_index]['data']['is_base_model_correct'] == info_list[max_index]['data']['is_jury_approving']:
            max_acc_judgements += 1

        def model_correct_poll(condition):
            if condition:
                if ll_info['is_base_model_correct']:
                    return 1
            elif not ll_info['is_base_model_correct']:
                    return 1
            return 0
        
        accurate_judgements += model_correct_poll(true_counts > false_counts)
        cal_acc_judgements += model_correct_poll(correct_score > incorrect_score)
        cal_mul_acc_judgements += model_correct_poll(correct_mul_score < incorrect_mul_score)

    print("Accurate vs Inaccurate judgements")
    print("Majority Poll:", accurate_judgements, total_disagreements - accurate_judgements)
    print("Calibrated Confidence Poll:", cal_acc_judgements, total_disagreements - cal_acc_judgements)
    print("Calibrated Mul Confidence Poll:", cal_mul_acc_judgements, total_disagreements - cal_mul_acc_judgements)
    print("Calibrated Max Poll:", max_cal_acc_judgements, total_disagreements - max_cal_acc_judgements)
    print("Max Poll (Uncalibrated):", max_acc_judgements, total_disagreements - max_acc_judgements)
    print("Veto Poll:", veto_acc_judgements, total_disagreements - veto_acc_judgements)
    print("Majorities:", true_majority, false_majority)
    print("True Majority:", true_majority_true, true_majority_false)
    print("False Majority:", false_majority_true, false_majority_false)
    print("Start Interval:", interval_start)
    print("End Interval:", interval_end)
    print("Llama correct:", jury_correct[0], len(llama_test_vals) - jury_correct[0], jury_correct[0]/total_disagreements)
    print("Olmo correct:", jury_correct[1], len(llama_test_vals) - jury_correct[1], jury_correct[1]/total_disagreements)
    print("Stable correct:", jury_correct[2], len(llama_test_vals) - jury_correct[2], jury_correct[2]/total_disagreements)
    print("Base Model Correct:", base_model_correct, len(llama_test_vals) - base_model_correct)
    '''
    on test data:
    Majority Poll: 153 16
    Calibrated Confidence Poll: 153 16
    Calibrated Mul Confidence Poll: 156 13
    Calibrated Max Poll: 158 11
    Veto Poll: 161 8
    Majorities: 18 151
    True Majority: 5 13
    False Majority: 3 148
    '''
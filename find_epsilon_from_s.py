def find_epsilon_from_s(s_value: float, alpha_dict: dict):
    for i in range(1, 100):
        eps = i/100
        if (1 - s_value) <= alpha_dict[eps] and s_value <= alpha_dict[eps] :
            continue
        return eps
    return 1
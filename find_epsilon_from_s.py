def find_epsilon_from_s(s_value: float, alpha_dict: dict) -> float:
    """
    Find the epsilon value corresponding to a given s_value based on the alpha_dict.
    :param s_value: The s_value for which to find the epsilon.
    :param alpha_dict: A dictionary mapping alpha values to their corresponding quantiles.
    :return: The epsilon value that corresponds to the given s_value.
    """
    for i in range(1, 100):
        eps = i/100
        if (1 - s_value) <= alpha_dict[eps] and s_value <= alpha_dict[eps] :
            continue
        return eps
    return 1
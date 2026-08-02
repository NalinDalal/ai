def raise_to_power(base_num, pow_num):
    """

    :param base_num: param pow_num:
    :param pow_num: 

    """
    result = 1
    for index in range(pow_num):
        result *= base_num
    return result

print(raise_to_power(2, 3))


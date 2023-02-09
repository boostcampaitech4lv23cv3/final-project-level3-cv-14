def score(user_solution=None, answer=None):
    """채점하는 함수

    Args:
        user_solution (dict): key : 문제번호, value : 정답 값
        answer (dict): key : 문제번호, value : 정답 값

    Returns:
        dict: key : 문제번호, value : O or X
    """
    result = {}
    for question in range(1, len(answer) + 1):
        if question not in user_solution.keys() or user_solution[question] == "-1":
            result[question] = "No"
        else:
            if user_solution[question] == answer[question]:
                result[question] = "O"
            else:
                result[question] = "X"
    return result

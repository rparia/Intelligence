
def compute_surprise(pred, outcome):
    stdout_diff = abs(len(pred["predicted_stdout"]) - len(outcome["stdout"]))
    exit_diff = abs(pred["predicted_exit_code"] - outcome["exit_code"])

    return (stdout_diff + exit_diff) * pred["confidence"]


import subprocess
import time

def execute(command: str):
    start = time.time()
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    stdout, _ = proc.communicate()
    end = time.time()

    return {
        "stdout": stdout.strip(),
        "exit_code": proc.returncode,
        "duration": end - start
    }

import requests
import subprocess
import json
import re

MODEL = "starcoder2:7b"
FILENAME = "Hello.java"
CLASSNAME = "Hello"

def generate_code(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        # force more deterministic output
        "options": {"temperature": 0.0, "top_p": 0.9}
    }

    response = requests.post(url, json=payload)
    data = response.json()
    code = data.get("response", "")

    # Print raw code from model
    print("Raw code from model:\n", code, "\n")

    # Remove common wrappers like ```java ... ``` or [source]
    code = re.sub(r"```.*?```", "", code, flags=re.DOTALL)
    code = re.sub(r"\[source.*?\]", "", code, flags=re.DOTALL)
    code = code.replace("<|dot|>", ".").replace("<|tab|>", "\t")

    # Strip stray tokens
    code = re.sub(r"<\|.*?\|>", "", code)

    # Print cleaned code
    print("Cleaned code:\n", code, "\n")

    return code.strip()


def write_and_compile(code):
    with open(FILENAME, "w") as f:
        f.write(code)

    with open("compile_errors.txt", "w") as err_file:
        compile_proc = subprocess.run(
            ["javac", FILENAME],
            stdout=subprocess.PIPE,
            stderr=err_file,
            text=True
        )

    return compile_proc

def run_program():
    run_proc = subprocess.run(["java", CLASSNAME], capture_output=True, text=True)
    return run_proc


# --- Main Loop ---
prompt = """
Write a complete Java program with class Hello that prints Hello Pranta. 
Output only valid Java code. 
Do not include markdown fences, explanations, or special tokens or [source].
"""

# for i in range(3):  # attempt up to 3 iterations
for i in range(1):
    print(f"\n=== Iteration {i+1} ===")
    code = generate_code(prompt)
    print("Generated code:\n", code)

    # compile_result = write_and_compile(code)
    # if compile_result.returncode != 0:
    #    print("Compilation failed:\n", compile_result.stderr)
    #    prompt = f"The code failed to compile with this error:\n{compile_result.stderr}\nFix it. Ensure the class is named {CLASSNAME}."
    #    continue

    # run_result = run_program()
    # if run_result.returncode != 0:
    #    print("Runtime error:\n", run_result.stderr)
    #    prompt = f"The code ran but crashed with this error:\n{run_result.stderr}\nFix it. Ensure the class is named {CLASSNAME}."
    #    continue

    # print("Program output:\n", run_result.stdout)
    break




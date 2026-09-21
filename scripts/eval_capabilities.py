#!/usr/bin/env python3
"""
Small, self-contained capability evaluation against a running SGLang OpenAI
server.

The subsets are deliberately tiny and deterministic so that the same questions
can be run against BF16 and Enhanced-W8A8 without downloading datasets.  They
are labeled ``*_subset`` in the output to avoid implying a full benchmark run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests


GSM8K_SUBSET = [
    ("A train travels 60 km/h for 2.5 hours and 80 km/h for 1.5 hours. What is the average speed?", 67.5),
    ("If 5 shirts cost $120, how much do 8 shirts cost?", 192),
    ("A rectangle has length 12 and width 7. What is its area?", 84),
    ("A car uses 8 liters per 100 km. How many liters for 250 km?", 20),
    ("What is 15% of 240?", 36),
    ("The sum of two numbers is 41 and their difference is 5. Find the larger number.", 23),
    ("A book has 360 pages. If Mia reads 24 pages per day, how many days does she need?", 15),
    ("A bakery sold 240 croissants in 6 hours. What was the hourly average?", 40),
    ("A box contains 4 red, 5 blue, and 6 green balls. How many balls total?", 15),
    ("If x + 7 = 19, what is x?", 12),
    ("A store gives a 20% discount on a $150 item. What is the sale price?", 120),
    ("How many minutes are there in 3 hours and 15 minutes?", 195),
    ("A number is tripled and then increased by 6 to get 33. Find the number.", 9),
    ("A triangle has a base of 10 and a height of 6. What is its area?", 30),
    ("If 3 notebooks cost $15, how much do 10 notebooks cost?", 50),
    ("A bus departs at 8:15 and arrives at 11:45. How many minutes is the trip?", 210),
    ("The product of two consecutive positive integers is 30. Find the larger one.", 6),
    ("A recipe needs 250 grams of flour for 4 servings. How much for 10 servings?", 625),
    ("A TV costs $480 after a 25% discount. What was the original price?", 640),
    ("What is the average of 18, 24, 30, 36, and 42?", 30),
]


MMLU_SUBSET = [
    ("Which planet is closest to the Sun?", ["Venus", "Earth", "Mercury", "Mars"], "C"),
    ("What is the chemical symbol for gold?", ["Go", "Gd", "Au", "Ag"], "C"),
    ("In which country is the Great Barrier Reef located?", ["Australia", "Brazil", "Indonesia", "Mexico"], "A"),
    ("Who wrote 'Romeo and Juliet'?", ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"], "B"),
    ("What is the largest ocean on Earth?", ["Atlantic", "Indian", "Arctic", "Pacific"], "D"),
    ("The Pythagorean theorem applies to which type of triangle?", ["equilateral", "right", "isosceles", "scalene"], "B"),
    ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], "C"),
    ("Which gas do plants primarily absorb during photosynthesis?", ["oxygen", "carbon dioxide", "nitrogen", "hydrogen"], "B"),
    ("What is 2 to the power of 10?", ["256", "512", "1024", "2048"], "C"),
    ("Which language is primarily used for Android app development?", ["Swift", "Kotlin", "C#", "Rust"], "B"),
    ("Who developed the theory of general relativity?", ["Isaac Newton", "Albert Einstein", "Niels Bohr", "Galileo Galilei"], "B"),
    ("What is the smallest prime number?", ["0", "1", "2", "3"], "C"),
    ("The powerhouse of the cell is the...", ["nucleus", "ribosome", "mitochondrion", "golgi apparatus"], "C"),
    ("Which element has atomic number 1?", ["helium", "hydrogen", "oxygen", "carbon"], "B"),
    ("A byte consists of how many bits?", ["4", "8", "16", "32"], "B"),
    ("Which country hosted the 2016 Summer Olympics?", ["China", "United Kingdom", "Brazil", "Japan"], "C"),
    ("What is the main currency of the United Kingdom?", ["euro", "dollar", "pound", "yen"], "C"),
    ("In computer science, FIFO describes a...", ["stack", "queue", "tree", "graph"], "B"),
    ("What is the derivative of x^2?", ["x", "2x", "x^2", "2"], "B"),
    ("Which blood type is the universal donor?", ["A", "B", "AB", "O"], "D"),
]


HUMANEVAL_SUBSET = [
    {
        "name": "sum_even_numbers",
        "prompt": "Write a Python function sum_even_numbers(numbers) that returns the sum of even integers in the input list.",
        "signature": "def sum_even_numbers(numbers):\n",
        "tests": [
            "assert sum_even_numbers([1,2,3,4]) == 6",
            "assert sum_even_numbers([]) == 0",
            "assert sum_even_numbers([-2,1,3,6]) == 4",
        ],
    },
    {
        "name": "reverse_string",
        "prompt": "Write a Python function reverse_string(s) that returns the reversed string.",
        "signature": "def reverse_string(s):\n",
        "tests": [
            "assert reverse_string('abc') == 'cba'",
            "assert reverse_string('') == ''",
            "assert reverse_string('hello world') == 'dlrow olleh'",
        ],
    },
    {
        "name": "is_palindrome",
        "prompt": "Write a Python function is_palindrome(s) that returns True if s is a palindrome ignoring spaces and case.",
        "signature": "def is_palindrome(s):\n",
        "tests": [
            "assert is_palindrome('A man a plan a canal Panama') is True",
            "assert is_palindrome('abc') is False",
            "assert is_palindrome('racecar') is True",
        ],
    },
    {
        "name": "count_words",
        "prompt": "Write a Python function count_words(text) that returns the number of whitespace-separated words.",
        "signature": "def count_words(text):\n",
        "tests": [
            "assert count_words('hello world') == 2",
            "assert count_words('') == 0",
            "assert count_words(' a  b c ') == 3",
        ],
    },
    {
        "name": "fibonacci",
        "prompt": "Write a Python function fibonacci(n) that returns the n-th Fibonacci number with F(0)=0, F(1)=1.",
        "signature": "def fibonacci(n):\n",
        "tests": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(10) == 55",
        ],
    },
]


def chat(url, prompt, max_tokens=256, temperature=0.0):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": "none",
    }
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def numeric_answer(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def choice_answer(text):
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else None


def eval_code(text, signature, tests):
    code = text
    block = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if block:
        code = block.group(1)
    # Prefer the first occurrence of the signature.
    start = code.find(signature.strip())
    if start >= 0:
        code = code[start:]
    script = code + "\n\n" + "\n".join(tests) + "\nprint('OK')\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return proc.returncode == 0 and "OK" in proc.stdout
    except Exception:
        return False


def run_gsm8k(url):
    correct = 0
    for q, answer in GSM8K_SUBSET:
        out = chat(url, f"{q}\nAnswer with only the final number.")
        pred = numeric_answer(out)
        if pred is not None and abs(pred - answer) < 1e-6:
            correct += 1
    return correct / len(GSM8K_SUBSET)


def run_mmlu(url):
    correct = 0
    for q, options, answer in MMLU_SUBSET:
        letters = ["A", "B", "C", "D"]
        choices = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        out = chat(url, f"{q}\n{choices}\nAnswer with only the letter.")
        pred = choice_answer(out)
        if pred == answer:
            correct += 1
    return correct / len(MMLU_SUBSET)


def run_humaneval(url):
    passed = 0
    for item in HUMANEVAL_SUBSET:
        out = chat(
            url,
            f"{item['prompt']}\nOnly output Python code starting with:\n"
            f"{item['signature']}",
            max_tokens=512,
        )
        if eval_code(out, item["signature"], item["tests"]):
            passed += 1
    return passed / len(HUMANEVAL_SUBSET)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model-name", default="Qwen3.5-27B-Enhanced-W8A8")
    parser.add_argument(
        "--output",
        default="/home/acceleration/quantization/benchmarks/capabilities.json",
    )
    args = parser.parse_args()

    results = {"model_name": args.model_name}
    for name, fn in [
        ("gsm8k_subset", run_gsm8k),
        ("mmlu_subset", run_mmlu),
        ("humaneval_subset", run_humaneval),
    ]:
        t = time.time()
        score = fn(args.url)
        results[name] = score
        print(f"{name}: {score:.4f} ({time.time() - t:.1f}s)")

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        all_results = json.loads(path.read_text())
    else:
        all_results = {}
    all_results[args.model_name] = results
    path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()

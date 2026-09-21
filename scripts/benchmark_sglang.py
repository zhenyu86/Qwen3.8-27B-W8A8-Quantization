#!/usr/bin/env python3
"""
Benchmark SGLang server performance.
Measures TTFT, TPOT, throughput, and memory for different concurrency levels.
"""

import argparse
import json
import os
import re
import time
import subprocess
import requests
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class BenchmarkResult:
    model_name: str
    quant_method: str
    concurrency: int
    prompt_length: int
    output_length: int
    ttft_ms: float          # Time to first token (ms)
    tpot_ms: float          # Time per output token (ms)
    total_time_s: float     # Total request time (s)
    output_tokens: int      # Tokens generated
    tokens_per_second: float # Output tokens/s for this request
    total_throughput: float  # Total throughput across all concurrent requests


def generate_prompt(length: int, style: str = "english") -> str:
    """Generate a prompt of approximate token length."""
    # Rough estimate: 1 token ≈ 4 chars for English, 2 chars for Chinese
    char_length = length * 4

    if style == "english":
        base = ("The quick brown fox jumps over the lazy dog. "
                "Artificial intelligence has made remarkable progress in recent years. "
                "Large language models can now understand and generate human-like text. "
                "Machine learning algorithms continue to improve across various benchmarks. ")
    elif style == "chinese":
        base = ("人工智能技术在过去几年取得了显著进展。大语言模型现在能够理解和生成类似人类的文本。"
                "机器学习算法在各种基准测试中持续改进。深度学习模型的量化技术是部署大语言模型的关键。")
    elif style == "code":
        base = ("def fibonacci(n):\n    if n <= 1:\n        return n\n"
                "    a, b = 0, 1\n    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n    return b\n\n")
    else:
        base = ("Explain the concept of attention mechanisms in neural networks. "
                "How do they enable models to focus on relevant parts of the input? ")

    repetitions = (char_length // len(base)) + 1
    prompt = (base * repetitions)[:char_length]
    return prompt


def send_request(
    url: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict:
    """Send a single inference request and measure timing."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    ttft = None
    first_token_time = None
    token_times = []
    all_tokens = []
    start_time = time.time()

    try:
        response = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=300,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get('choices', [])
                    if choices:
                        delta = choices[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            current_time = time.time()
                            if first_token_time is None:
                                first_token_time = current_time
                                ttft = (current_time - start_time) * 1000
                            token_times.append(current_time)
                            all_tokens.append(content)
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        return {
            'request_id': request_id,
            'error': str(e),
            'ttft_ms': 0,
            'tpot_ms': 0,
            'total_time_s': 0,
            'output_tokens': 0,
            'tokens_per_second': 0,
        }

    total_time = time.time() - start_time
    num_tokens = len(all_tokens)

    # Calculate TPOT (time between tokens)
    tpot = 0
    if len(token_times) > 1:
        inter_token_times = [token_times[i] - token_times[i-1]
                           for i in range(1, len(token_times))]
        tpot = statistics.mean(inter_token_times) * 1000 if inter_token_times else 0

    tps = num_tokens / total_time if total_time > 0 else 0

    return {
        'request_id': request_id,
        'ttft_ms': ttft or 0,
        'tpot_ms': tpot,
        'total_time_s': total_time,
        'output_tokens': num_tokens,
        'tokens_per_second': tps,
    }


def run_benchmark(
    url: str,
    model_name: str,
    quant_method: str,
    concurrency: int,
    prompt_length: int,
    output_length: int,
    num_runs: int = 3,
) -> BenchmarkResult:
    """Run benchmark with specified concurrency."""
    print(f"\n--- Benchmark: {model_name} | quant={quant_method} | "
          f"concurrency={concurrency} | prompt={prompt_length} | "
          f"output={output_length} ---")

    all_results = []

    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...")
        prompts = [generate_prompt(prompt_length) for _ in range(concurrency)]

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i, prompt in enumerate(prompts):
                future = executor.submit(send_request, url, prompt, output_length, i)
                futures.append(future)

            run_results = []
            for future in as_completed(futures):
                result = future.result()
                if 'error' not in result:
                    run_results.append(result)

        if run_results:
            all_results.extend(run_results)

    if not all_results:
        print("  WARNING: All requests failed!")
        return BenchmarkResult(
            model_name=model_name, quant_method=quant_method,
            concurrency=concurrency, prompt_length=prompt_length,
            output_length=output_length,
            ttft_ms=0, tpot_ms=0, total_time_s=0,
            output_tokens=0, tokens_per_second=0, total_throughput=0,
        )

    # Aggregate results
    avg_ttft = statistics.mean([r['ttft_ms'] for r in all_results])
    avg_tpot = statistics.mean([r['tpot_ms'] for r in all_results if r['tpot_ms'] > 0] or [0])
    avg_tps = statistics.mean([r['tokens_per_second'] for r in all_results])
    total_throughput = sum(r['tokens_per_second'] for r in all_results)
    avg_tokens = statistics.mean([r['output_tokens'] for r in all_results])

    result = BenchmarkResult(
        model_name=model_name,
        quant_method=quant_method,
        concurrency=concurrency,
        prompt_length=prompt_length,
        output_length=output_length,
        ttft_ms=avg_ttft,
        tpot_ms=avg_tpot,
        total_time_s=statistics.mean([r['total_time_s'] for r in all_results]),
        output_tokens=int(avg_tokens),
        tokens_per_second=avg_tps,
        total_throughput=total_throughput,
    )

    print(f"  Results: TTFT={result.ttft_ms:.1f}ms, TPOT={result.tpot_ms:.1f}ms, "
          f"throughput={result.total_throughput:.1f} tok/s, "
          f"tokens={result.output_tokens}")

    return result


def get_gpu_memory() -> float:
    """Get current GPU memory usage in GB."""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--json'],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        used = _sum_used_bytes(data)
        if used > 0:
            return used / (1024**3)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            values = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(values) / 1024  # MB to GB
    except Exception:
        pass

    return 0.0


def _sum_used_bytes(obj) -> float:
    """Recursively sum numeric values whose key mentions 'used'."""
    total = 0.0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                total += _sum_used_bytes(value)
            elif "used" in str(key).lower():
                total += _parse_bytes(value)
    elif isinstance(obj, list):
        for value in obj:
            total += _sum_used_bytes(value)
    return total


def _parse_bytes(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group())
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark SGLang Server")
    parser.add_argument("--url", type=str, default="http://localhost:30000",
                        help="SGLang server URL")
    parser.add_argument("--model-name", type=str, default="Qwen3.5-27B",
                        help="Model name for reporting")
    parser.add_argument("--quant-method", type=str, default="bf16",
                        help="Quantization method name")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 2048, 8192],
                        help="Prompt lengths to test")
    parser.add_argument("--output-length", type=int, default=256,
                        help="Output length in tokens")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 4, 8],
                        help="Concurrency levels to test")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs per configuration")
    parser.add_argument("--output-csv", type=str,
                        default="/home/acceleration/quantization/benchmarks/results.csv",
                        help="Output CSV file")
    parser.add_argument("--gpu-memory", type=float, default=0,
                        help="GPU memory in GB (manual override)")
    args = parser.parse_args()

    # Check server is running
    try:
        r = requests.get(f"{args.url}/v1/models", timeout=5)
        print(f"Server running: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {args.url}: {e}")
        return

    # Get GPU memory
    if args.gpu_memory > 0:
        gpu_mem = args.gpu_memory
    else:
        gpu_mem = get_gpu_memory()
    print(f"GPU Memory: {gpu_mem:.2f} GB")

    # Run benchmarks
    results = []
    for prompt_len in args.prompt_lengths:
        for conc in args.concurrency_levels:
            result = run_benchmark(
                url=args.url,
                model_name=args.model_name,
                quant_method=args.quant_method,
                concurrency=conc,
                prompt_length=prompt_len,
                output_length=args.output_length,
                num_runs=args.num_runs,
            )
            result_dict = asdict(result)
            result_dict['gpu_memory_gb'] = gpu_mem
            results.append(result_dict)

    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Append to existing CSV or create new
    fieldnames = [
        'model_name', 'quant_method', 'concurrency', 'prompt_length',
        'output_length', 'ttft_ms', 'tpot_ms', 'total_time_s',
        'output_tokens', 'tokens_per_second', 'total_throughput',
        'gpu_memory_gb',
    ]

    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\nResults saved to {args.output_csv}")
    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['quant_method']} | prompt={r['prompt_length']} | "
              f"conc={r['concurrency']} | "
              f"TTFT={r['ttft_ms']:.1f}ms | TPOT={r['tpot_ms']:.1f}ms | "
              f"throughput={r['total_throughput']:.1f} tok/s")


if __name__ == "__main__":
    main()

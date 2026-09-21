#!/usr/bin/env python3
"""检查 Qwen3.8-27B W8A8 量化项目的运行环境。

用法::

    python scripts/check_env.py

脚本只做只读检查，不会安装、升级或修改任何东西。
退出码 0 表示环境可用，1 表示存在阻塞项。
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BF16_MODEL = Path("/home/weight/Qwen3.8-27B")
QUANT_MODEL = Path("/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8")

CALIB_DATA = PROJECT_ROOT / "calibration" / "calib_data.jsonl"
TEST_DATA = PROJECT_ROOT / "calibration" / "test_data.jsonl"

# (导入名, 分发名, 最低版本)。版本按 “本项目实测版本” 取下限，
# torch / torchvision / triton / sglang 由 Hygon DTK 环境提供，见文档。
PACKAGES = [
    ("torch", "torch", "2.10.0"),
    ("transformers", "transformers", "5.6.0"),
    ("safetensors", "safetensors", "0.8.0"),
    ("accelerate", "accelerate", "1.14.0"),
    ("datasets", "datasets", "4.8.4"),
    ("compressed_tensors", "compressed-tensors", "0.15.0"),
    ("sglang", "sglang", "0.5.12"),
    ("requests", "requests", None),
    ("numpy", "numpy", None),
]

_NUMBERS = re.compile(r"\d+")


def _version_parts(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _NUMBERS.findall(version)[:3])


def _section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def check_packages(problems: list[str], warnings: list[str]) -> None:
    _section("Python 依赖")
    print(f"{'包名':<20} {'版本':<28} {'状态'}")
    print("-" * 68)

    for module_name, dist_name, min_version in PACKAGES:
        try:
            module = __import__(module_name)
        except Exception as exc:  # noqa: BLE001 - 需要把所有导入错误都报出来
            print(f"{dist_name:<20} {'-':<28} 未安装")
            problems.append(f"{dist_name} 无法导入: {exc!r}")
            continue

        version = getattr(module, "__version__", None)
        if not version:
            try:
                version = importlib_metadata.version(dist_name)
            except importlib_metadata.PackageNotFoundError:
                version = "unknown"

        status = "OK"
        if min_version is not None:
            try:
                if _version_parts(version) < _version_parts(min_version):
                    status = f"低于要求 {min_version}"
                    warnings.append(f"{dist_name} {version} < 要求 {min_version}")
            except ValueError:
                status = "版本无法解析"
        print(f"{dist_name:<20} {version:<28} {status}")


def check_gpu(problems: list[str], warnings: list[str]) -> None:
    _section("GPU (HIP / Hygon K100AI)")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"无法导入 torch: {exc!r}")
        problems.append("torch 不可用，无法检查 GPU")
        return

    print(f"PyTorch 版本   : {torch.__version__}")
    print(f"HIP 版本       : {torch.version.hip}")
    print(f"CUDA 版本      : {torch.version.cuda}")

    if torch.version.hip is None:
        warnings.append(
            "当前 torch 没有 HIP 支持，这不是 Hygon DTK 环境，"
            "请检查是否装成了别的 PyTorch 版本"
        )

    if not torch.cuda.is_available():
        print("设备可用性     : 不可用")
        problems.append("torch.cuda.is_available() 为 False，请检查 DTK / 驱动")
        return

    device_count = torch.cuda.device_count()
    print(f"设备可用性     : 可用，共 {device_count} 张卡")

    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        total = getattr(props, "total_memory", None)
        if total is None:  # 旧版 torch 的属性名
            total = getattr(props, "total_mem", 0)
        print(f"  - cuda:{index} {props.name} {total / 1024**3:.1f} GB")

    if device_count < 4:
        warnings.append(
            f"只检测到 {device_count} 张卡，本项目推理用 TP=4（4,5,6,7），"
            "卡数不足时只能改小 --tp-size"
        )

    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    print(f"HIP_VISIBLE_DEVICES : {visible if visible else '未设置（所有卡可见）'}")


def check_model_dir(path: Path, label: str, problems: list[str]) -> None:
    if not path.is_dir():
        problems.append(f"{label}不存在: {path}")
        print(f"{label:<12}: 不存在 ({path})")
        return

    has_config = (path / "config.json").is_file()
    shards = sorted(path.glob("*.safetensors"))
    has_index = (path / "model.safetensors.index.json").is_file()
    detail = f"{len(shards)} 个 safetensors 分片"
    if not has_index and len(shards) == 0:
        problems.append(f"{label}目录下没有权重分片: {path}")
    if not has_config:
        problems.append(f"{label}缺少 config.json: {path}")
    print(f"{label:<12}: {path}  (config={has_config}, {detail})")


def check_assets(problems: list[str]) -> None:
    _section("模型与数据")

    check_model_dir(BF16_MODEL, "BF16 模型", problems)
    check_model_dir(QUANT_MODEL, "W8A8 模型", problems)

    config_path = QUANT_MODEL / "config.json"
    if config_path.is_file():
        import json

        try:
            config = json.loads(config_path.read_text())
            quant_config = config.get("quantization_config", {})
            method = quant_config.get("quant_method")
            print(f"{'量化方式':<12}: {method}")
            if method != "w8a8_int8":
                problems.append(
                    f"量化配置的 quant_method 是 {method!r}，SGLang 需要 'w8a8_int8'；"
                    "启动时还要显式加 --quantization w8a8_int8"
                )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"无法解析 {config_path}: {exc!r}")

    for path, label in ((CALIB_DATA, "校准数据"), (TEST_DATA, "测试数据")):
        if path.is_file():
            print(f"{label:<12}: {path} ({_count_lines(path)} 条)")
        else:
            problems.append(f"{label}缺失: {path}")
            print(f"{label:<12}: 缺失 ({path})")

    for target in (QUANT_MODEL.parent, BF16_MODEL.parent):
        if target.is_dir():
            usage = shutil.disk_usage(target)
            free_gb = usage.free / 1024**3
            print(f"磁盘 {str(target):<32}: 剩余 {free_gb:.1f} GB")
            if free_gb < 60:
                problems.append(f"{target} 剩余空间不足 60 GB，量化写不下")


def main() -> int:
    problems: list[str] = []
    warnings: list[str] = []

    print("Qwen3.8-27B W8A8 量化项目 - 环境检查")
    print(f"项目目录 : {PROJECT_ROOT}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"解释器   : {sys.executable}")

    if sys.version_info[:2] != (3, 10):
        warnings.append(
            f"当前 Python 是 {sys.version_info.major}.{sys.version_info.minor}，"
            "本项目实测环境是 3.10"
        )

    check_packages(problems, warnings)
    check_gpu(problems, warnings)
    check_assets(problems)

    _section("检查结果")
    for item in warnings:
        print(f"[警告] {item}")
    for item in problems:
        print(f"[错误] {item}")

    if problems:
        print(f"\n存在 {len(problems)} 个阻塞项，请按上面的提示修复。")
        return 1

    print("\n环境检查通过，可以运行 scripts/quantize.py 和 SGLang 服务。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

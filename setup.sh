#!/usr/bin/env bash
#
# 环境检查 / 依赖补齐脚本。
#
#   ./setup.sh            只做检查，不修改任何东西（默认）
#   ./setup.sh --install  安装 requirements.txt 里缺失的依赖
#                         （不会碰 torch / torchvision / triton / sglang / DTK）
#
# 详细说明见 docs/SETUP.md。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DO_INSTALL=0

usage() {
  cat <<'EOF'
环境检查 / 依赖补齐脚本。

  ./setup.sh            只做检查，不修改任何东西（默认）
  ./setup.sh --install  安装 requirements.txt 里缺失的依赖

详细说明见 docs/SETUP.md。
EOF
}

for arg in "$@"; do
  case "$arg" in
    --install) DO_INSTALL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $arg（可用: --install, --help）" >&2; exit 2 ;;
  esac
done

echo "=== Qwen3.8-27B W8A8 量化项目 - 环境准备 ==="
echo "项目目录 : ${PROJECT_ROOT}"

echo
echo "--- 解释器与驱动 ---"
echo "Python   : $("${PYTHON}" -c 'import sys; print(sys.version.split()[0], sys.executable)')"

if command -v rocm-smi >/dev/null 2>&1; then
  echo "rocm-smi : $(command -v rocm-smi)"
elif [ -x /opt/dtk/bin/rocm-smi ]; then
  echo "rocm-smi : /opt/dtk/bin/rocm-smi（不在 PATH 中）"
else
  echo "rocm-smi : 未找到，请确认 DTK 已安装"
fi

if command -v hy-smi >/dev/null 2>&1; then
  echo "hy-smi   : $(command -v hy-smi)"
elif [ -x /opt/hyhal/bin/hy-smi ]; then
  echo "hy-smi   : /opt/hyhal/bin/hy-smi（不在 PATH 中）"
fi

if [ "$DO_INSTALL" -eq 1 ]; then
  echo
  echo "--- 安装运行时依赖 (requirements.txt) ---"
  "${PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
else
  echo
  echo "（默认只检查不安装。需要补齐依赖时运行: ./setup.sh --install）"
fi

echo
echo "--- 环境自检 ---"
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_env.py"

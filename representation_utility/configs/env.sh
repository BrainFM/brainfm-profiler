# Source this before running scripts.
#   bash:  source representation_utility/configs/env.sh
#   zsh:   source representation_utility/configs/env.sh
if [ -n "${BASH_SOURCE:-}" ]; then _src="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then _src="${(%):-%x}"
else _src="$0"; fi
HERE="$(cd "$(dirname "$_src")/.." && pwd)"
unset _src

export VIRTUAL_ENV="$HERE/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
# keep all model downloads inside the experiment folder
export HF_HOME="$HERE/weights/hf_cache"
export TORCH_HOME="$HERE/weights/torch_cache"
# load optional secrets (HF_TOKEN=...) if present
[ -f "$HERE/.env" ] && set -a && . "$HERE/.env" && set +a

echo "env ready: python=$(command -v python)  HF_HOME=$HF_HOME  HF_TOKEN=${HF_TOKEN:+set}"

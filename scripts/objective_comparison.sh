#!/bin/bash
# Frozen, sequential three-objective pilot on one host/GPU. No submission side effects.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/run.sh"
if [[ "${1:---help}" == --help ]]; then
  echo 'Usage: bash scripts/objective_comparison.sh --output experiments/<campaign> [common run.sh arguments]'
  echo 'Runs EIG, MSC and MOCU sequentially from a frozen committed source, continuing independent objectives after a failure.'
  exit 0
fi
[[ "${1:-}" == --output && -n "${2:-}" ]] || { echo 'Provide --output first' >&2; exit 2; }
CAMPAIGN="$2"
shift 2
[[ "$CAMPAIGN" == /* ]] || CAMPAIGN="$ROOT/$CAMPAIGN"
for arg in "$@"; do
  case "$arg" in
    --config|--config=*|--objective|--objective=*|--experiment_type|--experiment-type|--output)
      echo 'Campaign owns config/objective/output; provide only common settings.' >&2; exit 2 ;;
  esac
done
[[ ! -e "$CAMPAIGN/source" ]] || { echo 'Refusing to reuse a campaign source' >&2; exit 2; }
git -C "$ROOT" diff --quiet HEAD -- src configs scripts tools hprc run.sh sweep_run.sh AGENTS.md README.md requirements.txt || {
  echo 'Commit the reviewed source before freezing this campaign' >&2; exit 2;
}
mkdir -p "$CAMPAIGN/source" "$CAMPAIGN/logs"
git -C "$ROOT" rev-parse HEAD > "$CAMPAIGN/source_commit.txt"
git -C "$ROOT" archive HEAD src configs scripts tools hprc run.sh sweep_run.sh AGENTS.md README.md requirements.txt .gitignore |
  tar -xf - -C "$CAMPAIGN/source"
printf '%s\n' "$@" > "$CAMPAIGN/common_arguments.txt"
overall=0
for objective in eig msc mocu; do
  config=ieee9_mocu
  [[ "$objective" != eig ]] || config=ieee9_eig
  printf '%s\n' "$objective" > "$CAMPAIGN/current_objective.txt"
  echo "Starting $objective at $(date -Is)"
  if bash "$CAMPAIGN/source/run.sh" --config "$CAMPAIGN/source/configs/$config.yaml" \
      --objective "$objective" --output "$CAMPAIGN/$objective" "$@" \
      > "$CAMPAIGN/logs/$objective.log" 2>&1; then
    printf '%s\t%s\n' "$objective" completed >> "$CAMPAIGN/status.tsv"
  else
    result=$?
    printf '%s\tfailed:%s\n' "$objective" "$result" >> "$CAMPAIGN/status.tsv"
    overall=1
  fi
done
printf '%s\n' "$overall" > "$CAMPAIGN/exit_code"
exit "$overall"

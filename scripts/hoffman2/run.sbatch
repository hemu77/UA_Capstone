#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --job-name=ua-main-896
#SBATCH --no-requeue
set -euo pipefail
umask 077
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit through Slurm.'; exit 2; }
work=$(realpath -- "${1:?Pass the dedicated workspace, containing repo and env}")
[[ -d "$work/repo" && -x "$work/env/bin/python" ]] || exit 2
export HOME="$work/home" TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache"
export MPLCONFIGDIR="$work/cache/matplotlib" MPLBACKEND=Agg PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
mkdir -p "$HOME" "$TMPDIR" "$MPLCONFIGDIR" "$work/results"
# One coordinator across setup/relaunch mistakes, in addition to the runner lock.
exec 9>"$work/collector.lock"
flock -n 9 || { echo 'Another collector owns this workspace.'; exit 2; }
key="$work/private/openai.key"
[[ -f "$key" && ! -L "$key" && -O "$key" && $(stat -c %a "$key") == 600 ]] || {
    echo 'A private, owner-only API key file is required outside the repository.'; exit 2;
}
IFS= read -r OPENAI_API_KEY < "$key" || [[ -n "$OPENAI_API_KEY" ]]
[[ -n "$OPENAI_API_KEY" ]] || exit 2
export OPENAI_API_KEY
cd "$work/repo"
python="$work/env/bin/python"
set +e
"$python" -X utf8 -B -m final_study launch --stage main --approve-usd 120 \
    --approval-reference owner-eight-repetitions-120 --auto-recover --walltime-seconds 82800
code=$?
set -e
unset OPENAI_API_KEY
if [[ $code != 0 && $code != 75 ]]; then
    echo "Collector stopped with exit $code. No automatic scheduler retry."
    exit "$code"
fi
# The collector has released its lock; archive durable evidence, never credentials.
archive="$work/results/checkpoint-$SLURM_JOB_ID.tgz"
tar -czf "$archive.tmp" outputs/final_study final_study/contract.json
mv -- "$archive.tmp" "$archive"
if [[ $code == 75 ]]; then
    # afterok prevents overlap even if the continuation is immediately eligible.
    next=$(sbatch --parsable --account="$SLURM_JOB_ACCOUNT" --export=NONE \
        --dependency="afterok:$SLURM_JOB_ID" --chdir="$work/repo" \
        --output="$work/logs/main.%j.log" "$work/repo/scripts/hoffman2/run.sbatch" "$work")
    printf 'PAUSED_AT_CHECKPOINT; continuation_job=%s\n' "$next"
    exit 0
fi
"$python" -X utf8 -B -m final_study verify --stage main
"$python" -X utf8 -B -m final_study analyze --stage main
tar -czf "$work/results/final-verified.tgz.tmp" outputs/final_study final_study/contract.json
mv -- "$work/results/final-verified.tgz.tmp" "$work/results/final-verified.tgz"
echo 'FULL_COLLECTION_VERIFIED; retrieve final-verified.tgz and inspect terminal failures separately.'

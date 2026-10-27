#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=ua-storage
#SBATCH --no-requeue
set -euo pipefail
umask 077
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
work=$(realpath -- "${1:?Pass dedicated workspace}")
printf '{"status":"VALIDATION_PENDING"}\n' > "$work/storage-validation.json"
# A new directory prevents an earlier probe's signals from satisfying this run.
probe=$(mktemp -d "$work/storage-$SLURM_JOB_ID-XXXXXX")
unset OPENAI_API_KEY PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    "$work/env/bin/python" -X utf8 -B "$work/repo/scripts/hoffman2/verify_storage.py" "$probe"
cp -- "$probe/verified.json" "$work/storage-validation.json"

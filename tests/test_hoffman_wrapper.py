"""Exercise scheduler branches without a cluster, credentials, or model calls.

flock/stat are stand-ins here; the actual shared filesystem needs a remote probe.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = (str(Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/bin/bash.exe')
        if os.name == 'nt' else shutil.which('bash'))


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'Bash is required for scheduler tests')
class HoffmanWrapperTests(unittest.TestCase):
    def test_scheduler_restarts_are_disabled(self):
        wrapper = (ROOT/'scripts/hoffman2/run.sbatch').read_text()
        self.assertIn('#SBATCH --no-requeue\n', wrapper)

    def exercise(self, *, key=b'fixture-not-a-real-key\n', launch=0, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root/'work'
            for name in ('bin', 'work/private', 'work/env/bin', 'work/logs',
                         'work/repo/outputs/final_study', 'work/repo/final_study'):
                (root/name).mkdir(parents=True, exist_ok=True)
            (work/'private/openai.key').write_bytes(key)
            (work/'private/openai.key').chmod(0o600)
            (work/'repo/final_study/contract.json').write_text('{}')
            (work/'repo/outputs/final_study/evidence.json').write_text('{}')
            scripts = {
                'bin/flock': 'exit "${LOCK_EXIT:-0}"',
                'bin/stat': 'printf "600\\n"',
                'bin/sbatch': '''[[ -z ${OPENAI_API_KEY:-} ]] || exit 90
printf 'sbatch %s\\n' "$*" >> "$TEST_LOG"
[[ ${SUBMIT_EXIT:-0} == 0 ]] || exit "$SUBMIT_EXIT"
echo 9876''',
                'work/env/bin/python': '''[[ -z ${PYTHONOPTIMIZE:-} ]] || exit 96
printf 'python %s\\n' "$*" >> "$TEST_LOG"
case " $* " in
  *' launch '*) [[ -n ${OPENAI_API_KEY:-} ]] || exit 91; exit "$LAUNCH_EXIT";;
  *' verify '*) exit "${VERIFY_EXIT:-0}";;
  *' analyze '*) exit "${ANALYZE_EXIT:-0}";;
  *) exit 92;;
esac''',
            }
            if overrides.pop('ARCHIVE_FAIL', False):
                scripts['bin/tar'] = 'exit 93'
            for name, body in scripts.items():
                path = root/name
                path.write_text('#!/usr/bin/env bash\n'+body+'\n', encoding='utf8')
                path.chmod(0o700)
            env = dict(os.environ, SLURM_JOB_ID='1234', SLURM_JOB_ACCOUNT='fixture',
                       LAUNCH_EXIT=str(launch), **{k: str(v) for k, v in overrides.items()})
            env.pop('OPENAI_API_KEY', None)
            result = subprocess.run([BASH, '-c',
                'cd "$1"; export PATH="$PWD/bin:$PATH" TEST_LOG="$PWD/events.log"; '
                'exec bash "$2" "$PWD/work"', 'fixture', root.as_posix(),
                (ROOT/'scripts/hoffman2/run.sbatch').as_posix()],
                capture_output=True, text=True, env=env, timeout=20)
            log = (root/'events.log').read_text() if (root/'events.log').exists() else ''
            archives = {}
            for path in (work/'results').glob('*.tgz'):
                with tarfile.open(path) as archive:
                    archives[path.name] = archive.getnames()
            self.assertNotIn('fixture-not-a-real-key', result.stdout+result.stderr+log)
            self.assertTrue(all('private' not in name for names in archives.values() for name in names))
            return result, log, archives

    def test_key_without_final_newline_and_full_completion(self):
        result, log, archives = self.exercise(key=b'fixture-not-a-real-key')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('verify --stage main', log)
        self.assertIn('analyze --stage main', log)
        self.assertNotIn('sbatch', log)
        self.assertIn('final-verified.tgz', archives)

    def test_pause_retains_cap_and_schedules_only_after_success(self):
        result, log, archives = self.exercise(launch=75)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--approve-usd 120', log)
        self.assertIn('--walltime-seconds 82800', log)
        self.assertIn('--dependency=afterok:1234', log)
        self.assertIn('--export=NONE', log)
        self.assertEqual(sum(line.startswith('sbatch ') for line in log.splitlines()), 1)
        self.assertNotIn('verify --stage main', log)
        self.assertEqual(set(archives), {'checkpoint-1234.tgz'})

    def test_collector_error_never_requeues(self):
        for code in (1, 2, 137):
            with self.subTest(code=code):
                result, log, archives = self.exercise(launch=code)
                self.assertEqual(result.returncode, code)
                self.assertNotIn('sbatch', log)
                self.assertFalse(archives)

    def test_archive_or_submission_failure_stops(self):
        for setting, code in (('ARCHIVE_FAIL', 93), ('SUBMIT_EXIT', 94)):
            with self.subTest(setting=setting):
                result, log, _ = self.exercise(launch=75, **{setting: code})
                self.assertEqual(result.returncode, code)
                self.assertNotIn('continuation_job=', result.stdout)
                if setting == 'ARCHIVE_FAIL':
                    self.assertNotIn('sbatch', log)

    def test_verification_and_analysis_failure_do_not_claim_completion(self):
        for setting in ('VERIFY_EXIT', 'ANALYZE_EXIT'):
            with self.subTest(setting=setting):
                result, log, archives = self.exercise(**{setting: 95})
                self.assertEqual(result.returncode, 95)
                self.assertNotIn('FULL_COLLECTION_VERIFIED', result.stdout)
                self.assertNotIn('final-verified.tgz', archives)
                self.assertNotIn('sbatch', log)

    def test_lock_or_empty_key_stops_before_collector(self):
        for kwargs in ({'LOCK_EXIT': 1}, {'key': b''}):
            with self.subTest(kwargs=kwargs):
                result, log, archives = self.exercise(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(log, '')
                self.assertFalse(archives)

    def test_optimization_cannot_disable_python_assertions(self):
        result, _, _ = self.exercise(PYTHONOPTIMIZE='1')
        self.assertEqual(result.returncode, 0, result.stderr)

"""Primary rebuilds must never overwrite an existing release."""
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from unittest.mock import patch

from scripts import run_primary_evaluation_v2 as driver


class PrimaryRebuildTests(unittest.TestCase):
    def test_existing_release_is_untouched_before_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / 'outputs/primary_analysis/published'
            destination.mkdir(parents=True)
            seal = destination / 'evaluation_manifest.json'
            seal.write_text(json.dumps({'status': 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS'}))
            before = seal.read_bytes()
            with patch.object(driver, 'ROOT', root), \
                    patch.object(driver, 'run_core', side_effect=ValueError('missing fixture input')) as core:
                with self.assertRaises(ValueError):
                    driver.build(destination)
            self.assertEqual(seal.read_bytes(), before, 'An existing completed seal must not change')
            core.assert_not_called()

    def fixture(self, root):
        source = root/driver.SOURCE
        source.mkdir(parents=True)
        for name in ('summary.json', 'analysis_rows.json', 'analysis_summary.json',
                     'primary_contrasts.json', 'primary_contrasts.csv', 'network_metrics.csv',
                     'missingness_report.json'):
            (source/name).write_text('{}')
        code = root/'analysis_primary/core.py'
        code.parent.mkdir()
        code.write_text('# fixture analysis')
        script = root/'scripts/run_primary_evaluation_v2.py'
        script.parent.mkdir()
        script.write_text('# fixture driver')
        published = root/'outputs/primary_analysis/20261105'
        published.mkdir(parents=True)
        (published/'evaluation_manifest.json').write_text('{"status":"sealed fixture"}')
        return code, script, published

    def test_invalid_destinations_are_rejected_before_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, published = self.fixture(root)
            empty = published.parent/'empty'
            empty.mkdir()
            file = published.parent/'file'
            file.write_text('keep')
            for path in (root/'outside', published/'nested', published.parent, empty, file):
                with self.subTest(path=path), patch.object(driver, 'ROOT', root), \
                        patch.object(driver, 'run_core') as core:
                    with self.assertRaisesRegex(ValueError, 'NEW direct child'):
                        driver.build(path)
                    core.assert_not_called()
            self.assertEqual(list(empty.iterdir()), [])
            self.assertEqual(file.read_text(), 'keep')
            self.assertFalse((root/'outside').exists())
            self.assertFalse((published/'nested').exists())

    def test_output_argument_is_required(self):
        with patch.object(driver, 'build') as build, redirect_stdout(StringIO()), \
                patch('sys.stderr', new_callable=StringIO):
            with self.assertRaises(SystemExit) as error:
                driver.main([])
            self.assertEqual(error.exception.code, 2)
            build.assert_not_called()

    def test_missing_source_cannot_create_an_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root/'outputs/primary_analysis/reproduction'
            with patch.object(driver, 'ROOT', root), patch.object(driver, 'run_core') as core:
                with self.assertRaises(FileNotFoundError):
                    driver.build(destination)
                core.assert_not_called()
            self.assertFalse(destination.exists())

    def test_exclusive_creation_preserves_a_competing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, script, published = self.fixture(root)
            destination = published.parent/'new'
            mkdir = Path.mkdir

            def competing_creator(path, *args, **kwargs):
                if path == destination:
                    mkdir(path)
                    (path/'sentinel').write_text('keep')
                return mkdir(path, *args, **kwargs)

            with patch.object(driver, 'ROOT', root), patch.object(driver, '__file__', str(script)), \
                    patch.object(Path, 'mkdir', new=competing_creator), patch.object(driver, 'run_core') as core:
                with self.assertRaises(FileExistsError):
                    driver.build(destination)
                core.assert_not_called()
            self.assertEqual([p.name for p in destination.iterdir()], ['sentinel'])
            self.assertEqual((destination/'sentinel').read_text(), 'keep')

    def test_failure_marks_only_the_new_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, script, published = self.fixture(root)
            destination = published.parent/'failed'
            before = (published/'evaluation_manifest.json').read_bytes()
            with patch.object(driver, 'ROOT', root), patch.object(driver, '__file__', str(script)), \
                    patch.object(driver, 'run_core', side_effect=RuntimeError('missing receipts')):
                with self.assertRaisesRegex(RuntimeError, 'missing receipts'):
                    driver.build(destination)
            self.assertEqual(json.loads((destination/'evaluation_manifest.json').read_text())['status'],
                             'FAILED_INCOMPLETE')
            self.assertEqual((published/'evaluation_manifest.json').read_bytes(), before)

    def test_interrupt_marks_only_the_new_output_and_is_reraised(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, script, published = self.fixture(root)
            destination = published.parent/'interrupted'
            before = (published/'evaluation_manifest.json').read_bytes()
            with patch.object(driver, 'ROOT', root), patch.object(driver, '__file__', str(script)), \
                    patch.object(driver, 'run_core', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    driver.build(destination)
            self.assertEqual(json.loads((destination/'evaluation_manifest.json').read_text())['status'],
                             'FAILED_INCOMPLETE')
            self.assertEqual((published/'evaluation_manifest.json').read_bytes(), before)

    def pipeline(self, stack, root, script, destination, render=None, sources=None):
        data = object()
        stack.enter_context(patch.object(driver, 'ROOT', root))
        stack.enter_context(patch.object(driver, '__file__', str(script)))

        def core(actual_root, output):
            self.assertEqual((actual_root, output), (root, destination))
            (output/'result.csv').write_text('primary-only fixture')
            return {'sources': sources or {}}, data, {'contract_sha256': 'fixture', 'independent_numeric_crosscheck': True}

        stack.enter_context(patch.object(driver, 'run_core', side_effect=core))
        graph = stack.enter_context(patch.object(driver, 'run_graph_evaluation', return_value={
            'verified_graphs': 890, 'synthetic_graphs': 7120}))
        empirical = stack.enter_context(patch.object(driver, 'run_empirical_evaluation', return_value={
            'reference_networks': 8}))
        renderer = stack.enter_context(patch.object(driver, 'render_figures', side_effect=render,
                                                   return_value=[]))
        return data, graph, empirical, renderer

    def test_success_binds_code_and_outputs_without_modifying_published_release(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            _, script, published = self.fixture(root)
            destination = published.parent/'reproduction'
            before = (published/'evaluation_manifest.json').read_bytes()
            data, graph, empirical, renderer = self.pipeline(stack, root, script, destination)
            with redirect_stdout(StringIO()):
                result = driver.build(destination)
            for step in (graph, empirical, renderer):
                step.assert_called_once_with(root, data, destination)
            self.assertEqual(result['status'], 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS')
            self.assertEqual(result['runner_version'], 2)
            self.assertFalse(result['statistical_assumptions_verified'])
            self.assertEqual((result['supplemental_graphs_in_inference'], result['paid_calls'],
                              result['ledger_queries']), (0, 0, 0))
            self.assertEqual(result['code_sha256'][script.relative_to(root).as_posix()], driver.sha(script))
            self.assertEqual(result['artifacts']['result.csv']['sha256'], driver.sha(destination/'result.csv'))
            self.assertEqual((published/'evaluation_manifest.json').read_bytes(), before)

    def test_code_or_source_changes_refuse_a_completed_seal(self):
        for changed in ('analysis', 'driver', 'source'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                code, script, published = self.fixture(root)
                destination = published.parent/'reproduction'
                target = {'analysis': code, 'driver': script, 'source': root/driver.SOURCE/'summary.json'}[changed]

                def mutate(*args):
                    target.write_text('changed during computation')
                    return []

                self.pipeline(stack, root, script, destination, render=mutate)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    driver.build(destination)
                self.assertEqual(json.loads((destination/'evaluation_manifest.json').read_text())['status'],
                                 'FAILED_INCOMPLETE')

    def test_frozen_math_change_refuses_a_completed_seal(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            _, script, published = self.fixture(root)
            destination = published.parent/'reproduction'
            frozen = root/'final_study/analysis.py'
            frozen.parent.mkdir()
            frozen.write_text('# frozen statistical math')
            pins = {frozen.relative_to(root).as_posix(): driver.sha(frozen)}

            def mutate(*args):
                frozen.write_text('# changed during computation')
                return []

            self.pipeline(stack, root, script, destination, render=mutate, sources=pins)
            with self.assertRaisesRegex(ValueError, 'Frozen source changed: final_study/analysis.py'):
                driver.build(destination)
            self.assertEqual(json.loads((destination/'evaluation_manifest.json').read_text())['status'],
                             'FAILED_INCOMPLETE')


if __name__ == '__main__':
    unittest.main()

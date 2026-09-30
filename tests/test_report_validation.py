import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from adaptive_sorting.analysis.evaluation import EvaluationConfig
from adaptive_sorting.analysis.report_validation import (
    source_fingerprints, validate_cache, phase_plan, validate_job, validate_jobs,
)
from adaptive_sorting.analysis.summarize_agent_comparison import (
    METRICS_BY_RQ, paired_comparisons, supporting_definitions, MetricSummary, format_metric,
)


class ReportValidationTests(unittest.TestCase):
    def test_complete_report_renders_figures_and_can_be_selected(self):
        import csv
        from adaptive_sorting.analysis.generate_report import generate, select_current
        from adaptive_sorting.experiments.task_config import snapshot_task_config
        from adaptive_sorting.env.sorting_task_env import DEFAULT_TASK_RULES
        with TemporaryDirectory() as temp:
            root = Path(temp)
            trials = root / 'trials.csv'
            with trials.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['seed', 'trial', 'sphere_color', 'is_correct'])
                writer.writeheader()
                writer.writerows(dict(seed=seed, trial=i, sphere_color='red', is_correct=i % 5 != 0)
                                 for seed in (1, 2) for i in range(1, 301))
            snapshot = snapshot_task_config(DEFAULT_TASK_RULES, root)
            (root / 'metadata.json').write_text(json.dumps(dict(status='complete', seeds=[1, 2],
                                                               trials=300, task_config_snapshot=snapshot)))
            report = root / 'report'
            generate({'rq1': {'sources': {'A': str(trials)}}}, report)
            self.assertEqual(json.loads((report/'report_status.json').read_text())['status'], 'complete')
            index = json.loads((report/'figure_index.json').read_text())
            self.assertTrue(index)
            for figure in index.values():
                self.assertGreater((report/figure).stat().st_size, 0)
            self.assertTrue((report/'overview.png').is_file())
            select_current(report)
            self.assertEqual((root/'current').resolve(), report)

    def test_report_preflight_rejects_empty_or_mismatched_inputs(self):
        from adaptive_sorting.analysis.generate_report import generate, select_current
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'at least one'):
                generate({}, root/'empty')
            source = root/'trials.csv'
            source.write_text('unused')
            manifest = {'rq3a': {'sources': {'A': str(source)}},
                        'rq3b': {'sources': {'B': str(source)}}}
            with self.assertRaisesRegex(ValueError, 'matching controller'):
                generate(manifest, root/'mismatch')
            with self.assertRaisesRegex(ValueError, 'completed'):
                select_current(root)
            self.assertFalse((root/'empty').exists())
            self.assertFalse((root/'mismatch').exists())

    def test_statistical_hash_ignores_rendering_but_tracks_config_parser(self):
        import shutil
        from adaptive_sorting.analysis.report_validation import source_code_hashes, STATISTICAL_SOURCES
        source = Path(__file__).parents[1] / 'src/adaptive_sorting'
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in STATISTICAL_SOURCES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            renderer = root / 'analysis/plot_style.py'
            renderer.write_text('color = "red"')
            first = source_code_hashes(root)
            renderer.write_text('color = "blue"')
            second = source_code_hashes(root)
            self.assertEqual(first[0], second[0])
            self.assertNotEqual(first[1], second[1])
            parser = root / 'env/sorting_task_env.py'
            parser.write_text(parser.read_text() + '\n# parser revision\n')
            self.assertNotEqual(second[0], source_code_hashes(root)[0])

    def test_cache_detects_in_place_edits_and_statistical_settings(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'trials.csv'
            path.write_text('original')
            paths = [('A', path)]
            fingerprints = source_fingerprints(paths)
            config = EvaluationConfig()
            previous = {**config.metadata(), 'bootstrap_samples': 2000, 'bootstrap_seed': 0,
                        'binary_proportion_interval': 'wilson', 'paired_interval': 'seed_bootstrap',
                        'sources': {'A': str(path)}, 'source_fingerprints': fingerprints,
                        'analysis_source_sha256': 'code'}
            validate_cache(previous, paths, config, fingerprints, 'code', 2000, 0)
            for changed in ({'bootstrap_samples': 100}, {'analysis_source_sha256': 'old'},
                            {'source_fingerprints': None}):
                with self.assertRaisesRegex(ValueError, 'stale or unverifiable'):
                    validate_cache({**previous, **changed}, paths, config, fingerprints, 'code', 2000, 0)
            path.write_text('modified')
            with self.assertRaises(ValueError):
                validate_cache(previous, paths, config, source_fingerprints(paths), 'code', 2000, 0)
            path.write_text('original')
            path.with_name('metadata.json').write_text(json.dumps({'seeds': [1]}))
            with self.assertRaises(ValueError):
                validate_cache(previous, paths, config, source_fingerprints(paths), 'code', 2000, 0)

    def test_truncation_wrong_boundaries_and_missing_conditions_are_rejected(self):
        metadata = {'status': 'complete', 'seeds': [1, 2], 'change_sizes': [2, 4],
                    'before_trials': 2, 'after_trials': 3}
        plan = phase_plan('rq2a', metadata)
        rows = [dict(trial=i, phase='before_update' if i<=2 else 'after_update',
                     phase_trial=i if i<=2 else i-2) for i in range(1,6)]
        validate_job('rq2a', rows, plan)
        with self.assertRaisesRegex(ValueError, 'Trial count'):
            validate_job('rq2a', rows[:-1], plan)
        rows[2]['phase']='before_update'
        with self.assertRaisesRegex(ValueError, 'Phase boundaries'):
            validate_job('rq2a', rows, plan)
        validate_jobs('rq2a', [(1,2),(1,4),(2,2),(2,4)], metadata)
        with self.assertRaisesRegex(ValueError, 'Seed/condition'):
            validate_jobs('rq2a', [(1,2),(2,2)], metadata)
        with self.assertRaisesRegex(ValueError, 'completion'):
            phase_plan('rq2a', {**metadata, 'status': 'running'})

    def test_missing_final_recurring_phase_is_not_a_complete_run(self):
        plan=phase_plan('rq2b', {'initial_trials': 1, 'phase_trials': 2, 'phase_labels': ['A','B','A']})
        rows=[dict(trial=i, phase_trial=1 if i<3 else 2, phase_index=1 if i==1 else 2,
                   phase_label='A' if i==1 else 'B') for i in range(1,4)]
        with self.assertRaisesRegex(ValueError, 'Trial count'):
            validate_job('rq2b', rows, plan)

    def test_proportion_effects_use_percentage_points_and_shares_are_neutral(self):
        records={'A':[dict(rq='rq1', seed=i, final_new_task_accuracy=.9, new_task_errors=4) for i in (1,2)],
                 'B':[dict(rq='rq1', seed=i, final_new_task_accuracy=.85, new_task_errors=7) for i in (1,2)]}
        effects={r['metric']:r for r in paired_comparisons('rq1', records)}
        self.assertAlmostEqual(effects['final_new_task_accuracy']['mean_difference'],5)
        self.assertEqual(effects['final_new_task_accuracy']['unit'],'percentage_points')
        self.assertEqual(effects['new_task_errors']['mean_difference'],-3)
        self.assertEqual(effects['new_task_errors']['unit'],'errors')
        rule=next(d for d in METRICS_BY_RQ['rq4a'] if d.key.endswith('_share'))
        self.assertEqual(rule.direction,'neutral')
        self.assertTrue(all(d.direction=='neutral' for d in supporting_definitions('rq3a', [{'rule_relevant_only_share': .5}])))
        criterion=next(d for d in METRICS_BY_RQ['rq1'] if d.censored_median)
        self.assertEqual(format_metric(MetricSummary('A','scope',criterion,40,30,55,2,2,1200)),'40 [30, 55]')
        self.assertEqual(format_metric(MetricSummary('A','scope',criterion,1201,40,1201,1,3,1200)),'>1200 (1/3)')

    def test_report_cache_roundtrip_and_same_path_mutation(self):
        import csv
        from unittest.mock import patch
        from adaptive_sorting.analysis.generate_report import generate, select_current
        with TemporaryDirectory() as temp:
            root=Path(temp)
            trials=root/'trials.csv'
            rows=[dict(seed=seed, trial=i, phase_trial=i, sphere_color='red', is_correct=True)
                  for seed in (1,2) for i in range(1,301)]
            with trials.open('w') as stream:
                writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            trials.with_name('metadata.json').write_text(json.dumps({'status':'complete','seeds':[1,2],'trials':300}))
            manifest={'rq1': {'sources': {'A': str(trials)}}}
            with patch('adaptive_sorting.analysis.generate_report.curves.plot_rq1'), \
                 patch('adaptive_sorting.analysis.generate_report.curves.plot_learning_groups'), \
                 patch('adaptive_sorting.analysis.generate_report.summary.plot_summary'), \
                 patch('adaptive_sorting.analysis.generate_report.package_report'):
                generate(manifest, root/'first')
                from adaptive_sorting.analysis.report_validation import source_code_hashes
                statistical_hash, _ = source_code_hashes()
                (root/'first/rq1/core_metrics.md').write_text('obsolete presentation')
                with patch('adaptive_sorting.analysis.generate_report.source_code_hashes',
                           return_value=(statistical_hash, 'updated-figure-code')), \
                     patch('adaptive_sorting.analysis.generate_report.summary.evaluate_agents',
                           side_effect=AssertionError('Statistics must be reused')):
                    generate(manifest, root/'second', reuse_metrics=root/'first')
                self.assertNotEqual((root/'second/rq1/core_metrics.md').read_text(), 'obsolete presentation')
                self.assertEqual(json.loads((root/'first/report_status.json').read_text())['status'], 'complete')
                select_current(root/'first')
                with patch('adaptive_sorting.analysis.generate_report.curves.plot_rq1',
                           side_effect=RuntimeError('renderer failed')):
                    with self.assertRaisesRegex(RuntimeError, 'renderer failed'):
                        generate(manifest, root/'failed')
                failed = json.loads((root/'failed/report_status.json').read_text())
                self.assertEqual(failed['status'], 'failed')
                self.assertIn('renderer failed', failed['error'])
                with self.assertRaisesRegex(ValueError, 'completed'):
                    select_current(root/'failed')
                with self.assertRaisesRegex(ValueError, 'completed'):
                    generate(manifest, root/'bad-cache', reuse_metrics=root/'failed')
                self.assertFalse((root/'bad-cache').exists())
                self.assertEqual((root/'current').resolve(), root/'first')
                with self.assertRaisesRegex(ValueError, 'never current'):
                    generate(manifest, root/'current')
                select_current(root/'second')
                self.assertEqual((root/'current').resolve(), root/'second')
                (root/'current').unlink()
                (root/'current').mkdir()
                with self.assertRaisesRegex(ValueError, 'non-symlink'):
                    select_current(root/'first')
                a=json.loads((root/'first/rq1/evaluation_metadata.json').read_text())
                b=json.loads((root/'second/rq1/evaluation_metadata.json').read_text())
                self.assertEqual(a['analysis_source_sha256'], b['analysis_source_sha256'])
                self.assertNotEqual(a['figure_source_sha256'], b['figure_source_sha256'])
                self.assertEqual(a['source_fingerprints'], b['source_fingerprints'])
                self.assertEqual((root/'first/rq1/core_metrics.csv').read_bytes(), (root/'second/rq1/core_metrics.csv').read_bytes())
                trials.write_text(trials.read_text().replace('True','False',1))
                with self.assertRaisesRegex(ValueError,'source_fingerprints'):
                    generate(manifest, root/'third', reuse_metrics=root/'second')
                self.assertFalse((root/'third').exists())

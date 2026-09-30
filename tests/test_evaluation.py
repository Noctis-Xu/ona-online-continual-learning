"""Regression checks for the statistical definitions."""
import unittest

from adaptive_sorting.analysis.evaluation import EvaluationConfig, evaluate_seed
from adaptive_sorting.analysis.statistics import wilson_interval
from adaptive_sorting.analysis.summarize_agent_comparison import (
    METRICS_BY_RQ, paired_comparisons, summarize_records,
)


class EvaluationDefinitionTests(unittest.TestCase):
    def test_wilson_boundary_and_input_contract(self):
        mean, lower, upper = wilson_interval([1] * 200)
        self.assertEqual(mean, 1)
        self.assertAlmostEqual(lower, .9811546736)
        self.assertAlmostEqual(upper, 1)
        _, lower, upper = wilson_interval([0] * 200)
        self.assertAlmostEqual(lower, 0)
        self.assertAlmostEqual(upper, .0188453264)
        self.assertEqual(wilson_interval([]), (None, None, None))
        self.assertIsNotNone(wilson_interval([1])[1])
        with self.assertRaises(ValueError):
            wilson_interval([.5])

    def test_binary_dispatch_does_not_change_continuous_or_paired_intervals(self):
        records = [{"rq": "rq4a", "seed": seed,
                    "first_any_novel_accuracy": 1,
                    "first_any_novel_rule_share": 1,
                    "new_task_errors": 3, "old_task_errors": 0,
                    "final_new_task_accuracy": 1, "final_old_task_accuracy": 1,
                    "trials_to_criterion": 40, "criterion_reached": True, "criterion_horizon": 1200} for seed in range(200)]
        summaries = {s.metric.key: s for s in summarize_records("rq4a", {"A": records})}
        self.assertLess(summaries["first_any_novel_accuracy"].lower, 1)
        self.assertAlmostEqual(summaries["final_new_task_accuracy"].lower, 1)
        self.assertEqual(summaries["trials_to_criterion"].metric.ci_method, "seed_bootstrap_median")
        paired = paired_comparisons("rq4a", {"A": records, "B": records})
        self.assertTrue(all(r["ci_method"] == "seed_bootstrap" for r in paired))
        self.assertTrue(all(r["mean_difference"] == r["ci95_lower"] == r["ci95_upper"] == 0 for r in paired))

    def test_new_task_groups_filter_after_global_slice_and_seeds_have_equal_weight(self):
        def evaluate(seed, colors, correct):
            rows = [{"seed": seed, "trial": i, "phase_trial": i,
                     "phase": "expanded_inputs", "sphere_color": color,
                     "is_novel": color in ("indigo", "violet"), "is_correct": value}
                    for i, (color, value) in enumerate(zip(colors, correct), 1)]
            return evaluate_seed("rq4a", rows, EvaluationConfig(2, 2))[0]
        a = evaluate(1, ["red", "indigo", "violet", "indigo"], [False, True, False, True])
        b = evaluate(2, ["indigo", "violet", "red", "indigo"], [False, False, True, True])
        self.assertEqual((a["final_new_task_accuracy"], a["final_new_task_n"]), (.5, 2))
        self.assertEqual((b["final_old_task_accuracy"], b["final_old_task_n"]), (1, 1))
        self.assertEqual((a["new_task_errors"], a["old_task_errors"]), (1, 1))
        summary = next(s for s in summarize_records("rq4a", {"A": [a, b]}) if s.metric.key == "final_new_task_accuracy")
        self.assertEqual(summary.average, .75)  # A pooled estimate would be 2/3.
        supporting = {s.metric.key: s for s in summarize_records("rq4a", {"A": [a, b]}, supporting=True)}
        self.assertIn("final_group_indigo_accuracy", supporting)
        self.assertIn("final_group_violet_accuracy", supporting)
        self.assertEqual(supporting["first_indigo_accuracy"].metric.ci_method, "wilson")
        self.assertEqual(supporting["first_novel_accuracy"].metric.ci_method, "seed_bootstrap")
        self.assertIn("old_task_errors", {d.key for d in METRICS_BY_RQ["rq2a"]})
        self.assertNotIn("first_novel_accuracy", {d.key for d in METRICS_BY_RQ["rq4a"]})

class DiagnosticWindowTests(unittest.TestCase):
    def test_final_attribution_shares_the_behavior_window(self):
        from adaptive_sorting.analysis.rq3_relational_diagnostics import decision_attribution
        rows = []
        for seed in (1, 2):
            for trial in range(1, 601):
                rows.append(dict(seed=seed, trial=trial,
                    phase='off_only' if trial <= 200 else 'mixed_contexts',
                    phase_trial=trial if trial <= 200 else trial-200,
                    mixed_phase_start=201, mixed_phase_trials=400,
                    sphere_color='red', context='light_on' if trial > 200 else 'light_off',
                    is_correct=True, decision_source='learned_rule',
                    driving_rule_family='color+L1' if trial <= 300 else 'color+L1+L2'))
        config = EvaluationConfig()
        attribution = decision_attribution(rows, config)
        late = next(r for r in attribution if r['stage'] == 'final_mixed' and r['decision_category'] == 'relevant_with_l2')
        self.assertEqual(late['window_trials'], 300)
        self.assertEqual(late['trial_count'], 600)
        self.assertEqual(late['trial_share'], 1)
        self.assertEqual({r['stage'] for r in attribution}, {'final_mixed'})
        behavior = evaluate_seed('rq3b', rows[:600], config)[0]
        self.assertEqual(late['trial_share'], behavior['rule_relevant_with_l2_share'])
        rows[-1]['decision_source'] = ''
        incomplete = next(r for r in decision_attribution(rows, config)
                          if r['stage'] == 'final_mixed' and r['decision_category'] == 'relevant_with_l2')
        self.assertEqual(incomplete['trial_share'], '')
        self.assertEqual(incomplete['valid_seed_count'], 1)


if __name__ == '__main__':
    unittest.main()

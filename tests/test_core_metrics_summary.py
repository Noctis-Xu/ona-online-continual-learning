from __future__ import annotations
import csv
import math
from pathlib import Path
import random
import tempfile
import unittest

from adaptive_sorting.analysis.evaluation import (
    EvaluationConfig,
    evaluate_seed,
    rolling_accuracy,
)
from adaptive_sorting.analysis.statistics import bootstrap_interval
from adaptive_sorting.analysis.summarize_agent_comparison import (
    evaluate_agents,
    paired_comparisons,
    plot_summary,
    summarize_records,
    write_markdown_summary,
    write_summary_csv,
)
from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.env.sorting_task_env import DEFAULT_TASK_RULES
from adaptive_sorting.experiments.run_rq1 import run_seed
from adaptive_sorting.experiments.result_paths import write_experiment_results


def rows(values, **fields):
    return [{"seed": 1, "trial": i, "phase_trial": i, "sphere_color": "red", "is_correct": value, **fields} for i, value in enumerate(values, 1)]


class EvaluationTests(unittest.TestCase):
    def test_errors_continue_after_reaching_the_criterion(self):
        result = evaluate_seed("rq1", rows([True, True, False, False]), EvaluationConfig(2, 2, 1))[0]
        self.assertEqual(result["trials_to_criterion"], 2)
        self.assertEqual(result["new_task_errors"], 2)
        self.assertEqual(result["final_new_task_accuracy"], 0)

    def test_missed_criterion_is_censored_not_a_fabricated_time(self):
        result = evaluate_seed("rq1", rows([False]*4), EvaluationConfig(2, 2, 1))[0]
        self.assertIsNone(result["trials_to_criterion"])
        self.assertFalse(result["criterion_reached"])
        self.assertEqual(result["criterion_horizon"], 4)
        self.assertEqual(result["new_task_errors"], 4)

    def test_criterion_window_counts_only_new_task_observations(self):
        # Old-task trials are correct but must not fill the new-task window; time stays global.
        data = rows([True, False, True, True, True], phase="after_update", change_size=2)
        for row, changed in zip(data, [False, True, False, True, True]):
            row["is_changed"] = changed
            row["sphere_color"] = "red" if changed else "blue"
        result = evaluate_seed("rq2a", data, EvaluationConfig(2, 2, 1))[0]
        self.assertEqual(result["trials_to_criterion"], 5)
        self.assertEqual(result["new_task_errors"], 1)
        self.assertEqual(result["old_task_errors"], 0)

    def test_missing_expected_colors_are_not_zero_or_silently_dropped(self):
        result = evaluate_seed("rq1", rows([True]*4), EvaluationConfig(2, 2))[0]
        self.assertIsNone(result["final_group_blue_accuracy"])
        self.assertEqual(result["final_group_blue_n"], 0)
        self.assertIn("blue", result["missing_final_groups"])
        self.assertEqual(result["final_new_task_accuracy"], 1)

    def test_short_window_is_unavailable_not_resized(self):
        result = evaluate_seed("rq1", rows([True]*3), EvaluationConfig(4, 2))[0]
        self.assertFalse(result["final_window_complete"])
        self.assertIsNone(result["final_new_task_accuracy"])

    def test_boolean_csv_strings_do_not_treat_false_as_true(self):
        result = evaluate_seed("rq1", rows(["False", "True"]), EvaluationConfig(2, 2))[0]
        self.assertEqual(result["new_task_errors"], 1)
        self.assertEqual(result["final_new_task_accuracy"], .5)
        with self.assertRaises(ValueError):
            evaluate_seed("rq1", rows(["wrong"]), EvaluationConfig(1, 1))

    def test_global_window_is_sliced_before_target_filter(self):
        data = rows([False, True, True, False], phase="after_update", change_size=2, is_changed=True)
        data[-2]["is_changed"] = False
        data[-2]["sphere_color"] = "blue"
        data[-1]["is_changed"] = False
        data[-1]["sphere_color"] = "blue"
        result = evaluate_seed("rq2a", data, EvaluationConfig(2, 2))[0]
        self.assertIsNone(result["final_new_task_accuracy"])
        self.assertEqual(result["final_old_task_accuracy"], .5)
        self.assertEqual(result["new_task_errors"], 1)
        self.assertEqual(result["old_task_errors"], 1)
        self.assertEqual(result["final_new_task_n"], 0)

    def test_old_task_errors_cover_the_whole_evaluated_phase(self):
        data = rows([True, False, True, False], phase="before_update", change_size=2, is_changed=False)
        data += [{"seed":1, "trial":i+4, "phase_trial":i, "phase":"after_update", "change_size":2, "sphere_color":c, "is_correct":v, "is_changed":c=="red"} for i,(c,v) in enumerate([("blue", False),("red",False),("blue",True),("red",True)],1)]
        result=evaluate_seed("rq2a",data,EvaluationConfig(2,2))[0]
        self.assertEqual(result["old_task_errors"],1)
        self.assertEqual(result["new_task_errors"],1)
        self.assertEqual(result["phase_errors"],2)
        self.assertEqual(result["final_old_task_accuracy"],1)

    def test_recurring_phases_remain_separate(self):
        data=[]
        for phase in (1,2,3):
            for i in range(1,5):
                data.append({"seed":1,"trial":len(data)+1,"phase_index":phase,"phase_label":str(phase),"phase_trial":i,"sphere_color":"red" if i%2 else "blue","is_changed":i%2==1,"is_correct":phase!=2})
        results=evaluate_seed("rq2b",data,EvaluationConfig(2,2))
        self.assertEqual([r["phase_index"] for r in results],[1,2,3])
        self.assertEqual([r["new_task_errors"] for r in results],[0,2,0])
        summaries=summarize_records("rq2b",{"A":results},samples=100)
        self.assertEqual({s.scope for s in summaries},{"phase=2:2","phase=3:3"})
        self.assertTrue(all(s.n_expected==1 for s in summaries))

    def test_rq3_rule_categories_require_color_and_l1(self):
        data=rows([True]*4,phase="mixed_contexts",context="light_on",decision_source="learned_rule")
        for row,family in zip(data,["L1","color+L1","color+L1+L2","color+L2"]):
            row["driving_rule_family"]=family
        result=evaluate_seed("rq3b",data,EvaluationConfig(4,2))[0]
        self.assertEqual(result["rule_relevant_only_share"],.25)
        self.assertEqual(result["rule_relevant_with_l2_share"],.25)
        self.assertEqual(result["rule_missing_required_share"],.5)
        self.assertIsNone(result["final_old_task_accuracy"])

    def test_novel_first_encounters_track_feedback_and_joint_evidence(self):
        data=rows([True,False,True,True],phase="expanded_inputs",is_novel=True,decision_source="learned_rule",driving_rule_class="target_match_generalization")
        for row,color in zip(data,["indigo","indigo","violet","violet"]):
            row["sphere_color"]=color
        data[2]["driving_rule_class"]="grounded"
        result=evaluate_seed("rq4b",data,EvaluationConfig(2,2))[0]
        self.assertEqual(result["first_novel_accuracy"],1)
        self.assertEqual(result["first_novel_rule_share"],.5)
        self.assertEqual(result["first_novel_correct_rule_share"],.5)
        self.assertEqual(result["first_violet_prior_novel_feedback"],2)
        self.assertEqual(result["first_any_novel_prior_novel_feedback"],0)

    def test_missing_provenance_is_not_zero_rule_use(self):
        data=rows([True,True],phase="expanded_inputs",is_novel=True)
        data[0]["sphere_color"]="indigo"
        data[1]["sphere_color"]="violet"
        result=evaluate_seed("rq4a",data,EvaluationConfig(2,2))[0]
        self.assertEqual(result["first_novel_accuracy"],1)
        self.assertIsNone(result["first_novel_rule_share"])

    def test_curve_uses_available_observations_and_continues_across_transition(self):
        data=[(1,True),(1,True),(2,False),(2,False)]
        curve=rolling_accuracy(data,2,correct=lambda r:r[1])
        self.assertEqual(curve[0], 1)
        self.assertEqual(curve[2], .5)
        self.assertEqual(curve[1],1)
        self.assertEqual(curve[3],0)

    def test_curve_slices_global_window_before_filtering_and_keeps_empty_groups_missing(self):
        data = [(1, True, True), (1, False, False), (1, False, True),
                (2, True, False), (2, False, True), (2, True, False), (2, True, False)]
        values = rolling_accuracy(data, 2, correct=lambda r:r[1], include=lambda r:r[2])
        self.assertEqual(values[:3], [1, 1, 0])
        self.assertEqual(values[3], 0)
        self.assertEqual(values[4], 0)
        self.assertEqual(values[5], 0)
        self.assertTrue(math.isnan(values[6]))

    def test_missing_novel_color_does_not_become_a_complete_transfer_score(self):
        data = rows([True, True], phase="expanded_inputs", is_novel=True)
        for row in data:
            row["sphere_color"] = "indigo"
        result = evaluate_seed("rq4a", data, EvaluationConfig(2, 2))[0]
        self.assertEqual(result["first_indigo_accuracy"], 1)
        self.assertIsNone(result["first_violet_accuracy"])
        self.assertIsNone(result["first_novel_accuracy"])
        self.assertEqual(result["first_any_novel_accuracy"], 1)

    def test_duplicate_trials_are_rejected(self):
        data=rows([True]*2)
        data[1]["trial"]=1
        with self.assertRaises(ValueError):
            evaluate_seed("rq1",data,EvaluationConfig(2,2))


class ReportTests(unittest.TestCase):
    def test_runner_and_csv_evaluation_match(self):
        result,data=run_seed(3,300,300,DEFAULT_TASK_RULES,lambda actions, seed:UCB1Agent(actions,rng=random.Random(seed)))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            trial_path,summary_path=write_experiment_results(root,[result],data)
            records=evaluate_agents("rq1",[("UCB1",trial_path)],EvaluationConfig())
            self.assertEqual(records["UCB1"],[result])
            self.assertIn("metrics_version",summary_path.read_text())
            summaries=summarize_records("rq1",records,samples=100)
            write_summary_csv(root/'metrics.csv',summaries)
            write_markdown_summary(root/'metrics.md',"rq1",[("UCB1",trial_path)],summaries)
            plot_summary(root/'metrics.png',"rq1",[("UCB1",trial_path)],summaries)
            self.assertGreater((root/'metrics.png').stat().st_size,10000)
            self.assertNotIn("Observed best",(root/'metrics.md').read_text())

    def test_paired_differences_match_seed_ids_and_preserve_missing_counts(self):
        def record(seed,errors):
            return {"rq":"rq1","seed":seed,"new_task_errors":errors,"final_new_task_accuracy":1,
                    "trials_to_criterion":errors,"criterion_reached":True,"criterion_horizon":200}
        records={"A":[record(2,100),record(1,10)],"B":[record(1,15),record(2,105),record(3,20)]}
        paired=paired_comparisons("rq1",records,samples=100)
        self.assertNotIn('trials_to_criterion',{r['metric'] for r in paired})
        effect=next(r for r in paired if r['metric']=='new_task_errors')
        self.assertEqual(effect['mean_difference'],-5)
        self.assertEqual(effect['ci95_lower'],-5)
        self.assertEqual(effect['n_pairs'],2)
        self.assertEqual(effect['missing_seeds'],'3')
        summaries=summarize_records("rq1",records,samples=100)
        self.assertIsNone(next(s for s in summaries if s.agent=='A').average)

    def test_task_mismatch_rejected_even_when_seed_ids_match(self):
        with tempfile.TemporaryDirectory() as temp:
            paths=[]
            for label,color in [('A','red'),('B','blue')]:
                path=Path(temp)/f'{label}.csv'
                data=rows([True]*2)
                for row in data:row['sphere_color']=color
                with path.open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
                paths.append((label,path))
            with self.assertRaisesRegex(ValueError,'Unpaired input'):
                evaluate_agents('rq1',paths,EvaluationConfig(2,2))

    def test_context_control_allows_l2_only_but_checks_l1_and_controller(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for label, light2 in (("with L2", "on"), ("without L2", None)):
                folder = Path(temp) / label
                folder.mkdir()
                path = folder / "trials.csv"
                data = rows([True, False], phase="mixed_contexts", context="light_on")
                if light2:
                    for row in data:
                        row["light_2"] = light2
                with path.open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
                (folder / "metadata.json").write_text(json.dumps({"agent": "relational_ona", "seeds": [1]}))
                paths.append((label, path))
            with self.assertRaisesRegex(ValueError, "Unpaired input"):
                evaluate_agents("rq3b", paths, EvaluationConfig(2, 2))
            result = evaluate_agents("rq3b", paths, EvaluationConfig(2, 2), context_control=True)
            self.assertEqual(len(result), 2)
            second = paths[1][1]
            original = second.read_text()
            second.write_text(original.replace("light_on", "light_off"))
            with self.assertRaisesRegex(ValueError, "Unpaired input"):
                evaluate_agents("rq3b", paths, EvaluationConfig(2, 2), context_control=True)
            second.write_text(original)
            second.with_name("metadata.json").write_text(json.dumps({"agent": "ucb1", "seeds": [1]}))
            with self.assertRaisesRegex(ValueError, "same agent"):
                evaluate_agents("rq3b", paths, EvaluationConfig(2, 2), context_control=True)

    def test_rq2_csv_and_runner_share_stage_and_group_definitions(self):
        from adaptive_sorting.experiments.run_rq2a import run_seed as run_update
        config = EvaluationConfig(20)
        result, data = run_update(
            change_size=2, seed=5, before_trials=30, after_trials=40,
            final_window=20, config_path=DEFAULT_TASK_RULES,
            agent_factory=lambda actions, seed: UCB1Agent(actions, rng=random.Random(seed)),
        )
        with tempfile.TemporaryDirectory() as temp:
            path, _ = write_experiment_results(Path(temp), [result], data)
            offline = evaluate_agents("rq2a", [("UCB1", path)], config)
            self.assertEqual(offline["UCB1"], [result])
            self.assertEqual(result["phase_trials"], 40)
            self.assertEqual(result["new_task_errors"], sum(not r["is_correct"] for r in data if r["phase"] == "after_update" and r["is_changed"]))
            self.assertEqual(result["old_task_errors"], sum(not r["is_correct"] for r in data if r["phase"] == "after_update" and not r["is_changed"]))

    def test_incomplete_logged_seed_list_is_rejected_against_metadata(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / "trials.csv"
            data = rows([True, True])
            with path.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
            (folder / "metadata.json").write_text(json.dumps({"seeds": [1, 2]}))
            with self.assertRaisesRegex(ValueError, "disagree with metadata"):
                evaluate_agents("rq1", [("Agent", path)], EvaluationConfig(2, 2))

    def test_single_seed_has_no_estimated_confidence_interval(self):
        self.assertEqual(bootstrap_interval([.8]),(.8,None,None))

    def test_criterion_median_counts_missed_runs_as_beyond_the_phase(self):
        def record(seed, trial):
            return {"rq":"rq1","seed":seed,"trials_to_criterion":trial,"criterion_reached":trial is not None,"criterion_horizon":100,
                    "new_task_errors":1,"final_new_task_accuracy":1}
        mostly = summarize_records("rq1",{"A":[record(1,10),record(2,None),record(3,None)]},samples=100)
        result = next(s for s in mostly if s.metric.key=="trials_to_criterion")
        self.assertGreater(result.average,result.horizon)
        self.assertEqual((result.n_valid,result.n_expected),(1,3))
        half = summarize_records("rq1",{"A":[record(1,10),record(2,30),record(3,None)]},samples=100)
        self.assertEqual(next(s for s in half if s.metric.key=="trials_to_criterion").average,30)


if __name__=='__main__':
    unittest.main()

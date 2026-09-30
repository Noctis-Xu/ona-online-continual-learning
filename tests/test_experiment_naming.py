"""Check experiment IDs and controller labels."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from adaptive_sorting.experiments.agent_factory import agent_label
from adaptive_sorting.experiments.run_rq_comparison import parse_args
from adaptive_sorting.execution.ros2_experiment_demo import parse_args as parse_ros


class ExperimentNamingTests(unittest.TestCase):
    def test_ids_select_protocols_and_reject_unknown_ids(self):
        args = parse_args(['rq2a', 'rq3a', 'rq4a', 'flat_ona'])
        self.assertEqual(args.targets, ['rq2a', 'rq3a', 'rq4a'])
        for unknown in ('rq4', 'rq5'):
            with self.assertRaises(SystemExit):
                parse_args([unknown])

    def test_ros_public_ids_select_the_same_task(self):
        self.assertEqual(parse_ros(['rq4a']).rq, 'rq4a')
        self.assertEqual(parse_ros(['rq2a']).rq, 'rq2a')

    def test_tip_labels_use_verified_build_identity(self):
        with TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'custom-engine'
            binary.write_bytes(b'engine')
            manifest = {'variant': 'ona-tip', 'binary': {
                'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}}
            Path(str(binary)+'.profile.json').write_text(json.dumps(manifest))
            self.assertEqual(agent_label('flat_ona', binary), 'ONA-TIP (flat)')
            self.assertEqual(agent_label('relational_ona', binary), 'ONA-TIP (relational)')
            self.assertEqual(agent_label('ucb1', binary), 'Contextual UCB1')
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                agent_label('flat_ona', binary)

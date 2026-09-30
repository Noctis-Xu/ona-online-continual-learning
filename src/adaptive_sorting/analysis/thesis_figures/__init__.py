"""Result figures of the thesis (Chapter 4 and Appendix B).

The figures are drawn from the runs listed in report manifests (the format of
`adaptive_sorting.analysis.generate_report`), the implication traces of Experiment 2b,
and the runs at the raised suppression threshold. Each figure needs only some of
these inputs. Run `python -m adaptive_sorting.analysis.thesis_figures --help`.
"""
import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from adaptive_sorting.analysis.plot_style import plt

WINDOW = 50
MAIN_LABELS = ['ONA (flat)', 'ONA (relational)', 'Epsilon-greedy', 'Contextual UCB1', 'Contextual SW-UCB']
# Contrast hue for a modified ONA (ONA-TIP, a changed threshold) against ONA (relational), drawn solid.
CONTRAST_COLOR = '#E07B39'
TEXT_WIDTH = 15 / 2.54  # inches
PRINTED_FONT_SIZE = 6  # pt, the size of figure text on the page


@dataclass
class Inputs:
    """Runs from which the thesis results are computed."""

    manifest: Path | None = None
    extended_manifest: Path | None = None
    # Trace run directory per controller label, each holding evidence_trace.jsonl.
    traces: dict[str, Path] = field(default_factory=dict)
    # trials.csv of ONA (relational) at the suppression threshold 0.65 per experiment.
    threshold: dict[str, Path] = field(default_factory=dict)
    output_dir: Path = Path('thesis_figures')


INPUTS = Inputs()


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--manifest', type=Path,
                        help='report manifest of the main runs, with the ONA-TIP runs as ONA-TIP (relational)')
    parser.add_argument('--extended-manifest', type=Path,
                        help='report manifest of the runs of Experiments 3a and 3b with a mixed phase of 4800 trials')
    parser.add_argument('--trace', type=Path, help='implication trace directory of ONA (relational) in Experiment 2b')
    parser.add_argument('--tip-trace', type=Path, help='implication trace directory of ONA-TIP (relational) in Experiment 2b')
    for experiment in ('rq3a', 'rq3b'):
        parser.add_argument(f'--threshold-{experiment}', type=Path,
                            help=f'trials.csv of ONA (relational) at suppression threshold 0.65 in {experiment}')


def set_inputs(args: argparse.Namespace) -> None:
    INPUTS.manifest = args.manifest
    INPUTS.extended_manifest = args.extended_manifest
    INPUTS.traces = {label: path for label, path in (('ONA (relational)', args.trace),
                                                     ('ONA-TIP (relational)', args.tip_trace)) if path}
    INPUTS.threshold = {rq: path for rq in ('rq3a', 'rq3b') if (path := getattr(args, f'threshold_{rq}'))}


def require(value, option: str):
    if not value:
        raise SystemExit(f'{option} is required for this result')
    return value


def manifest(extended: bool = False) -> dict:
    """Load a report manifest with its trial paths resolved against the manifest's directory."""
    path = require(INPUTS.extended_manifest if extended else INPUTS.manifest,
                   '--extended-manifest' if extended else '--manifest')
    sources = json.loads(path.read_text())
    for spec in sources.values():
        spec['sources'] = {label: (path.parent / trials).resolve() for label, trials in spec['sources'].items()}
    return sources


def source(sources: dict, experiment: str, label: str) -> Path:
    try:
        return sources[experiment]['sources'][label]
    except KeyError:
        raise SystemExit(f'The manifest lists no {label} run for {experiment}') from None


def trace(label: str) -> Path:
    option = '--trace' if label == 'ONA (relational)' else '--tip-trace'
    return require(INPUTS.traces.get(label), option) / 'evidence_trace.jsonl'


def threshold_run(experiment: str) -> Path:
    return require(INPUTS.threshold.get(experiment), f'--threshold-{experiment}')


def output_path(name: str) -> Path:
    path = INPUTS.output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def printed(canvas_width: float, page_fraction: float = 1.0):
    """Draw figures so their text prints at PRINTED_FONT_SIZE; usable as a decorator or a `with` block.

    The canvases keep their report size (`canvas_width` inches, as in the figure's figsize) and are
    scaled to `page_fraction` of the text width, as in the `\\includegraphics` width in the thesis;
    only the font sizes follow from that scale, so curves and layout stay as in the reports.
    Tick labels are one step smaller so that every 200th trial fits in a third of the text width.
    """
    size = PRINTED_FONT_SIZE * canvas_width / (TEXT_WIDTH * page_fraction)
    with plt.rc_context({'font.size': size, 'legend.fontsize': .9 * size, 'axes.titlesize': 'medium',
                         'xtick.labelsize': .85 * size, 'ytick.labelsize': .85 * size}):
        yield

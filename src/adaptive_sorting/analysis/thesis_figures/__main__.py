"""Draw the result figures of the thesis; name figures to draw only those."""
import argparse
from pathlib import Path

from adaptive_sorting.analysis.thesis_figures import INPUTS, add_input_arguments, curves, mechanisms, set_inputs

FIGURES = {
    'rq1': curves.rq1,
    'rq2a': curves.rq2a,
    'rq2b': curves.rq2b,
    'rq2b_tip': curves.rq2b_tip,
    'rq2b_tip_mechanism': mechanisms.tip_mechanism,
    'rq3': curves.rq3,
    'rq3_decision_sources': mechanisms.decision_sources,
    'rq3_extended': curves.rq3_extended,
    'rq3_babbling_threshold': curves.babbling_threshold,
    'rq4': curves.rq4,
    'rq4_generalization_sources': mechanisms.generalization_sources,
    'tip_all_experiments': curves.tip_all_experiments,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figures', nargs='*', metavar='FIGURE', help=f'default: all of {", ".join(FIGURES)}')
    add_input_arguments(parser)
    parser.add_argument('--output-dir', type=Path, default=INPUTS.output_dir)
    args = parser.parse_args()
    names = args.figures or list(FIGURES)
    if unknown := set(names) - set(FIGURES):
        parser.error(f'unknown figures: {", ".join(sorted(unknown))}')
    set_inputs(args)
    INPUTS.output_dir = args.output_dir
    for name in names:
        FIGURES[name]()
        print(name)


if __name__ == '__main__':
    main()

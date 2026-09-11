#!/usr/bin/env python3
"""Plot only independently verified expert-monitoring smoke results."""

import argparse
import csv
import json
import os
from pathlib import Path


def main(root):
    report = json.loads((root / 'verification.json').read_text())
    assert report['verified'] and len(report['cases']) == 2
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    phases = ('code_only', 'reasoning', 'answer')
    labels = ('Reading the code', 'Reasoning', 'Final answer')
    matrices = {(c['filename'], p): np.zeros((43, 256)) for c in report['cases'] for p in phases}
    with (root / 'expert_activity.csv').open() as handle:
        for row in csv.DictReader(handle):
            key = (row['file'], row['phase'])
            if key in matrices:
                matrices[key][int(row['layer']), int(row['expert_id'])] = int(row['activations'])
    for case in report['cases']:
        for phase in phases:
            matrices[case['filename'], phase] /= case['phases'][phase]['tokens']
    minimum = min(m[m > 0].min() for m in matrices.values())
    maximum = max(1., max(m.max() for m in matrices.values()))
    norm = LogNorm(vmin=minimum, vmax=maximum)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e8edf3')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    fig.subplots_adjust(left=0.065, right=0.88, bottom=0.13, top=0.86, wspace=0.16, hspace=0.42)
    for row, case in enumerate(report['cases']):
        for col, (phase, label) in enumerate(zip(phases, labels)):
            axis = axes[row, col]
            values = matrices[case['filename'], phase]
            picture = axis.imshow(np.ma.masked_equal(values, 0), origin='upper', aspect='auto',
                                  interpolation='nearest', cmap=cmap, norm=norm)
            axis.axhline(2.5, color='white', linewidth=0.9, linestyle='--')
            axis.set_title(f'{label} | {case["phases"][phase]["tokens"]:,} tokens\n{case["filename"]} | CSV: {case["truth"]}',
                           fontsize=10, loc='left', pad=8)
            axis.set_xticks([0, 64, 128, 192, 255])
            axis.set_yticks([0, 10, 20, 30, 42])
            axis.set_xlabel('Logical expert ID')
            if col == 0:
                axis.set_ylabel('Layer')
    color_axis = fig.add_axes([0.91, 0.18, 0.015, 0.63])
    fig.colorbar(picture, cax=color_axis, label='Selections per token (log scale)')
    fig.suptitle('DeepSeek V4 Flash: measured expert activity', fontsize=20, x=0.065, ha='left', y=0.98)
    fig.text(0.065, 0.922, 'FreeToken live tracing | 43 layers x 256 routed experts | reasoning enabled', color='#475569', fontsize=12)
    fig.text(0.065, 0.045,
             'Gray = never selected in that phase. Dashed line separates hash-routed layers 0-2 from learned-routing layers 3-42.\n'
             'Reading panels show only source-code tokens. Every forward token selects six routes per layer; shared experts remain enabled.',
             fontsize=10, color='#475569')
    fig.savefig(root / 'expert-activity.png', dpi=180, facecolor='white')
    fig.savefig(root / 'expert-activity.svg', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir)

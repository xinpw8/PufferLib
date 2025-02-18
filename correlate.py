import pandas as pd
import numpy as np
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='puffer-pong-protein-lin')
parser.add_argument('--refresh', action='store_true', help='Refresh runs table')
parser.add_argument('--output_csv', type=str, default='sweep.csv', help='Output CSV file')
parser.add_argument('--output_json', type=str, default='sweep.json', help='Output JSON file')
parser.add_argument('--threshold', type=float, default=19, help='Score threshold')
args = parser.parse_args()

# Import and initialize Neptune project
if args.refresh or not os.path.exists(args.output_csv):
    import neptune
    project = neptune.init_project(
        project="pufferai/ablations",
        mode='read-only',
    )

    # Fetch runs table
    table = project.fetch_runs_table(
        tag=[args.tag],
        state='inactive'
    ).to_pandas()

    table.to_csv(args.output_csv, index=False)
else:
    table = pd.read_csv(args.output_csv)

table = table.sort_values(by='sys/creation_time')

# Correlation computation
correlations = {}
summary = {}
stats = {}
output_data = {}

scores = table['environment/score']
good_scores = scores[scores > args.threshold]

gamma = table['train/gamma']
summary['runs'] = len(scores)
summary['threshold'] = args.threshold
summary['num good'] = len(scores[scores > args.threshold])
summary['best'] = np.max(scores)
summary['mean good'] = np.mean(scores[scores > args.threshold])
summary['std good'] = np.std(scores[scores > args.threshold])

for key in table.columns:
    if not key.startswith('train/'):
        continue

    df = pd.DataFrame({
        'score': scores,
        'metric': table[key]
    }).dropna()

    try:
        corr = df['score'].corr(df['metric'])
    except Exception as e:
        continue

    if np.isnan(corr):
        continue

    correlations[key] = corr
    good_runs = df.loc[df['score'] > args.threshold, 'metric'].to_list()
    stats[key] = {
        'best': df.loc[df['score'] == np.max(df['score']), 'metric'].to_list()[0],
        'sensitivity': 1 / (np.log(np.max(good_runs)) - np.log(np.min(good_runs))),
        'min': np.min(good_runs),
        'max': np.max(good_runs),
        'mean': np.mean(good_runs),
        'std': np.std(good_runs),
    }

    # Append data for CSV
    output_data[key] = {
        'hparam': df['metric'].to_list(),
        'score': df['score'].to_list(),
    }


# Sort correlations
correlations = sorted(correlations.items(),
    key=lambda x: abs(x[1]), reverse=True)

# Print correlations
print('Correlations:')
for key, corr in correlations:
    print(f'\t{key}: {corr}')

print('Summary:')
for key, summ in summary.items():
    print(f'\t{key}: {summ}')

print()
for key, stats in stats.items():
    print(key)
    for k, v in stats.items():
        print(f'\t{k}: {v}')

# Save JSON
import json
with open(args.output_json, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f'Saved correlations to {args.output_json}')

scores = table['environment/score'].to_numpy()[:70]
costs = table['cost'].to_numpy()[:70]

pareto_scores = []
pareto_costs = []
eps = 1e-6
for idx, obs in enumerate(scores):
    cost = costs[idx]
    higher_score = scores + eps > scores[idx]
    lower_cost = costs - eps < costs[idx]
    better = higher_score & lower_cost
    better[idx] = False
    if not better.any():
        pareto_scores.append(obs)
        pareto_costs.append(cost)

from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.plotting import figure, show
from bokeh.palettes import Turbo256

# Create a ColumnDataSource that includes the 'order' for each point
source = ColumnDataSource(data=dict(
    x=costs,
    y=scores,
    order=list(range(len(scores)))  # index/order for each point
))

pareto_source = ColumnDataSource(data=dict(
    x=pareto_costs,
    y=pareto_scores,
    order=list(range(len(pareto_scores)))  # index/order for each point
))

# Define a color mapper across the range of point indices
mapper = LinearColorMapper(
    palette=Turbo256,
    low=0,
    high=len(scores)
)

# Set up the figure
p = figure(title='Visualize sweep', 
           x_axis_label='Cost', 
           y_axis_label='Score')

p.scatter(x='x', 
          y='y', 
          color={'field': 'order', 'transform': mapper}, 
          size=10, 
          source=source)

'''
p.line(x='x', 
       y='y', 
       color='purple',
       source=curve)
'''
show(p)

import argparse
import os
import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import Legend

def manyline(p, x, y, **kwargs):
    for i in range(y.shape[0]): p.line(x, y[i], line_width=1, line_alpha=0.5, **kwargs)

def stdevshade(p, x, y, **kwargs):
    μ = y.mean(0)
    σ = y.std(0) / np.sqrt(y.shape[0])
    p.varea(x, μ-σ, μ+σ, fill_alpha=.2, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='./results', help="root dir to save results")
    args = parser.parse_args()

    ## Linear regression warm-up
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Training steps', y_axis_label='1 - R^2', width=500, height=300, x_range=(10, 1e6), x_axis_type='log', y_axis_type='log')
        output_file(f'{args.root}/plot01.html')
        # p.add_layout(Legend(), 'left')
        results = np.load(f'{args.root}/experiment01/results.npz')
        stdevshade(p, results['steps'][0], 1-results['train'], legend_label="In-sample", fill_color='#129490')
        p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="In-sample", line_width=3, line_color='#129490', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="Out-of-sample", fill_color='#129490')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="Out-of-sample", line_width=3, line_color='#129490')
        p.legend.click_policy = "hide"
        p.legend.location = 'top_right'
        save(p)
    except FileNotFoundError: print("File missing for experiment01, skipping...")

    ## Linear regression warm-up, but with L2 loss
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Training steps', y_axis_label='1 - R^2', width=500, height=300, x_range=(100, 2e5),  y_range=(.5, 1), x_axis_type='log', y_axis_type='log')
        output_file(f'{args.root}/plot02a.html')
        # p.add_layout(Legend(), 'left')
        results = np.load(f'{args.root}/experiment01/results.npz')
        # stdevshade(p, results['steps'][0], results['train'], legend_label="Unregularized", fill_color='#129490')
        p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="Unregularized", line_width=2, line_color='#129490', line_dash='dashed')
        # stdevshade(p, results['steps'][0], results['test'], legend_label="Unregularized", fill_color='#129490')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="Unregularized", line_width=2, line_color='#129490')
        colors = ['#E0A890', '#AE621B', '#704E2E']
        for i, wd in enumerate([.05, .5, 1.]):
            results = np.load(f'{args.root}/experiment02/results{wd:06.2f}.npz')
            # stdevshade(p, results['steps'][0], results['train'], legend_label=f"L2={wd}", fill_color=colors[i])
            p.line(results['steps'][0], 1-results['train'].mean(0), legend_label=f"L2={wd}", line_width=2, line_color=colors[i], line_dash='dashed')
            # stdevshade(p, results['steps'][0], results['test'], legend_label=f"L2={wd}", fill_color=colors[i])
            p.line(results['steps'][0], 1-results['test'].mean(0), legend_label=f"L2={wd}", line_width=2, line_color=colors[i])
        p.legend.click_policy = "hide"
        p.legend.location = 'top_center'
        save(p)

        p = figure(border_fill_color='#f4f4f6', x_axis_label='Regularization strength', y_axis_label='1 - R^2', width=500, height=300, y_axis_type='log', x_range=(0, 1))
        output_file(f'{args.root}/plot02b.html')
        # p.add_layout(Legend(), 'left')
        x = [0., .05, .1, .2, .5, 1.]
        sources = [np.load(f'{args.root}/experiment01/results.npz')] + [np.load(f'{args.root}/experiment02/results{wd:06.2f}.npz') for wd in x[1:]]
        y_train = [1-src['train'].mean(0)[-1] for src in sources]
        y_test = [1-src['test'].mean(0)[-1] for src in sources]
        p.line(x, y_train, legend_label="In-sample", line_width=2, line_color='black', line_dash='dashed')
        p.line(x, y_test, legend_label="Out-of-sample", line_width=2, line_color='black', line_dash='solid')
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for experiment02, skipping...")

    ## Linear regression on bigger data
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Training steps', y_axis_label='1 - R^2', width=500, height=300, x_range=(10, 2e5), x_axis_type='log', y_axis_type='log')
        output_file(f'{args.root}/plot03.html')
        # p.add_layout(Legend(), 'left')
        results = np.load(f'{args.root}/experiment01/results.npz')
        stdevshade(p, results['steps'][0], 1-results['train'], legend_label="D=1000", fill_color='#129490')
        p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="D=1000", line_width=3, line_color='#129490', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="D=1000", fill_color='#129490')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="D=1000", line_width=3, line_color='#129490')
        results = np.load(f'{args.root}/experiment03/results.npz')
        stdevshade(p, results['steps'][0], 1-results['train'], legend_label="D=10000", fill_color='#065143')
        p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="D=10000", line_width=3, line_color='#065143', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="D=10000", fill_color='#065143')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="D=10000", line_width=3, line_color='#065143')
        p.legend.location = 'top_right'
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for experiment03, skipping...")

    ## Linear regression on various dataset sizes
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Dataset size', y_axis_label='1 - R^2', width=500, height=300, x_axis_type='log', x_range=(200,50000), y_axis_type='log', y_range=(0.3,2.))
        output_file(f'{args.root}/plot04.html')
        # p.add_layout(Legend(), 'left')
        dataset_sizes = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="In-sample", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="In-sample", line_width=3, line_color='#907AD6', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Out-of-sample", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Out-of-sample", line_width=3, line_color='#907AD6')
        p.legend.location = 'top_right'
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for experiment04, skipping...")

    ## Linear regression on various dataset sizes, augmenting input features with their squares
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Dataset size', y_axis_label='1 - R^2', width=500, height=300, x_axis_type='log', x_range=(200,50000), y_axis_type='log', y_range=(.3,2))
        output_file(f'{args.root}/plot05.html')
        # p.add_layout(Legend(), 'left')
        dataset_sizes = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Regular features", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Regular features", line_width=3, line_color='#907AD6', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Regular features", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Regular features", line_width=3, line_color='#907AD6')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment05/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Square features", fill_color='#4F518C')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Square features", line_width=3, line_color='#4F518C', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Square features", fill_color='#4F518C')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Square features", line_width=3, line_color='#4F518C')
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for experiment05, skipping...")

    ## Linear regression on various dataset sizes, augmenting input features with all pairwise products
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Dataset size', y_axis_label='1 - R^2', width=500, height=300, x_axis_type='log', x_range=(500,50000), y_axis_type='log', y_range=(.2,2))
        output_file(f'{args.root}/plot06.html')
        # p.add_layout(Legend(), 'left')
        dataset_sizes = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Regular features", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Regular features", line_width=3, line_color='#907AD6', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Regular features", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Regular features", line_width=3, line_color='#907AD6')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment05/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Square features", fill_color='#4F518C')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Square features", line_width=3, line_color='#4F518C', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Square features", fill_color='#4F518C')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Square features", line_width=3, line_color='#4F518C')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment06/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Pair-prod features", fill_color='#2C2A4A')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Pair-prod features", line_width=3, line_color='#2C2A4A', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Pair-prod features", fill_color='#2C2A4A')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Pair-prod features", line_width=3, line_color='#2C2A4A')
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for experiment06, skipping...")

    ## Deep learning on small data
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Training steps', y_axis_label='1 - R^2', width=500, height=300, x_axis_type='log', y_axis_type='log', x_range=(10,2e5), y_range=(.2,4.))
        output_file(f'{args.root}/plot07.html')
        # p.add_layout(Legend(), 'left')
        results = np.load(f'{args.root}/experiment07/results_linear.npz')
        stdevshade(p, results['steps'][0], 1-results['train'], legend_label="Linear regression", fill_color='#129490')
        p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="Linear regression", line_width=3, line_color='#129490', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="Linear regression", fill_color='#129490')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="Linear regression", line_width=3, line_color='#129490')
        results = np.load(f'{args.root}/experiment07/results_dl.npz')
        stdevshade(p, results['steps'][0], 1-results['train'], legend_label="Nonlinear regression", fill_color='#90BE6D')
        p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="Nonlinear regression", line_width=3, line_color='#90BE6D', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="Nonlinear regression", fill_color='#90BE6D')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="Nonlinear regression", line_width=3, line_color='#90BE6D')
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for experiment07, skipping...")

    ## Deep learning on various dataset sizes
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Dataset size', y_axis_label='1 - R^2', width=500, height=300, x_axis_type='log', x_range=(200,50000), y_axis_type='log', y_range=(.1,1.2))
        output_file(f'{args.root}/plot08.html')
        # p.add_layout(Legend(), 'left')
        dataset_sizes = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Regular features", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Regular features", line_width=3, line_color='#907AD6', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Regular features", fill_color='#907AD6')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Regular features", line_width=3, line_color='#907AD6')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment06/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Pair-prod features", fill_color='#2C2A4A')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Pair-prod features", line_width=3, line_color='#2C2A4A', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Pair-prod features", fill_color='#2C2A4A')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Pair-prod features", line_width=3, line_color='#2C2A4A')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment08/results{d:05}.npz')
            r2_train.append(results['train'][:,-1])
            r2_test.append(results['test'][:,-1])
        r2_train = np.stack(r2_train, 1)
        r2_test = np.stack(r2_test, 1)
        stdevshade(p, dataset_sizes, 1-r2_train, legend_label="Nonlinear regression", fill_color='#90BE6D')
        p.line(dataset_sizes, 1-r2_train.mean(0), legend_label="Nonlinear regression", line_width=3, line_color='#90BE6D', line_dash='dashed')
        stdevshade(p, dataset_sizes, 1-r2_test, legend_label="Nonlinear regression", fill_color='#90BE6D')
        p.line(dataset_sizes, 1-r2_test.mean(0), legend_label="Nonlinear regression", line_width=3, line_color='#90BE6D')
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for experiment08, skipping...")


    ## Deep learning at various model sizes on various dataset sizes
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Dataset size', y_axis_label='1 - R^2', width=500, height=300, x_axis_type='log', x_range=(500,50000), y_axis_type='log', y_range=(.005,2))
        output_file(f'{args.root}/plot09.html')
        # p.add_layout(Legend(), 'left')
        colors = ['#E138FF', '#A938EB', '#8B3DFA', '#5C38EB', '#3336FF', '#0A69FE']
        model_size_names = ['Small', 'Medium', 'Large', 'Huge', 'Enormous']
        dataset_sizes = [500, 1500, 5000, 15000, 50000]
        for i, m in enumerate(model_size_names):
            r2_train = []; r2_test = []
            for d in dataset_sizes:
                results = np.load(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz')
                r2_train.append(results['train'][:, -1])
                r2_test.append(results['test'][:, -1])
            r2_train = np.stack(r2_train, 1)
            r2_test = np.stack(r2_test, 1)
            stdevshade(p, dataset_sizes, 1-r2_train, legend_label=m, fill_color=colors[i])
            p.line(dataset_sizes, 1-r2_train.mean(0), legend_label=m, line_width=3, line_color=colors[i], line_dash='dashed')
            stdevshade(p, dataset_sizes, 1-r2_test, legend_label=m, fill_color=colors[i])
            p.line(dataset_sizes, 1-r2_test.mean(0), legend_label=m, line_width=3, line_color=colors[i])
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for plot09, skipping...")

    ## Deep learning at various model sizes on various dataset sizes
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Dataset size', y_axis_label='1 - R^2',  width=600, height=300, x_axis_type='log', y_axis_type='log', y_range=(.06,.5))
        output_file(f'{args.root}/plot10.html')
        p.add_layout(Legend(), 'left')
        colors = ['#E138FF', '#A938EB', '#8B3DFA', '#5C38EB', '#3336FF', '#0A69FE']
        model_size_names = ['Small', 'Medium', 'Large', 'Huge', 'Enormous']
        dataset_sizes = [500, 1500, 5000, 15000, 50000]
        for i, m in enumerate(model_size_names):
            r2_train = []; r2_test = []
            for d in dataset_sizes:
                results = np.load(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz')
                r2_train.append(results['train'][:, -1])
                r2_test.append(results['test'][:, -1])
            r2_train = np.stack(r2_train, 1)
            r2_test = np.stack(r2_test, 1)
            stdevshade(p, dataset_sizes, 1-r2_test, legend_label=m, fill_color=colors[i])
            p.line(dataset_sizes, 1-r2_test.mean(0), legend_label=m, line_width=2, line_color=colors[i])
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for plot10, skipping...")

    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Model size (parameter count)', y_axis_label='1 - R^2', width=600, height=300, x_axis_type='log', y_axis_type='log')
        output_file(f'{args.root}/plot11.html')
        p.add_layout(Legend(), 'left')
        colors = ['#B8E3FF', '#97C6E3', '#88B2CD', '#5E8CC2', '#3E6FA3', '#175485', '#003A68', '#001126']
        dataset_sizes = [500, 1500, 5000, 15000, 50000]
        model_shapes = [[32, 32, 16], [64, 64, 32], [128, 128, 64, 64], [256]*4, [512]*4]
        def ms_to_pc(ms): return sum(il*ol+ol for il, ol in zip([256]+ms, ms+[1]))
        model_pcs = [ms_to_pc(ms) for ms in model_shapes]
        for j, d in enumerate(dataset_sizes):
            r2_train = []; r2_test = []
            for i, m in enumerate(model_pcs):
                results = np.load(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz')
                r2_test.append(results['test'][:,-1])
            r2_test = np.stack(r2_test, 1)
            stdevshade(p, model_pcs, 1-r2_test, legend_label=f'D={d}', fill_color=colors[j])
            p.line(model_pcs, 1-r2_test.mean(0), legend_label=f'D={d}', line_width=3, line_color=colors[j])
        p.legend.click_policy = "hide"
        save(p)
    except FileNotFoundError: print("File missing for plot11, skipping...")

    ## Deep learning on small data, plus auxilliary data
    try:
        p = figure(border_fill_color='#f4f4f6', x_axis_label='Training steps', y_axis_label='1 - R^2', width=500, height=300, y_axis_type='log', x_range=(10, 2e5))
        output_file(f'{args.root}/plot12a.html')
        # p.add_layout(Legend(), 'left')
        results = np.load(f'{args.root}/experiment10/results_regular.npz')
        # stdevshade(p, results['steps'][0], 1-results['train'], legend_label="Original dataset", fill_color='#129490')
        # p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="Original dataset", line_width=3, line_color='#129490', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="Original dataset", fill_color='#129490')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="Original dataset", line_width=3, line_color='#129490')
        results = np.load(f'{args.root}/experiment10/results_aux.npz')
        # stdevshade(p, results['steps'][0], 1-results['train'], legend_label="Secondary dataset", fill_color='#90BE6D')
        # p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="Secondary dataset", line_width=3, line_color='#90BE6D', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="Secondary dataset", fill_color='#90BE6D')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="Secondary dataset", line_width=3, line_color='#90BE6D')
        results = np.load(f'{args.root}/experiment10/results_regular_with_aux.npz')
        # stdevshade(p, results['steps'][0], 1-results['train'], legend_label="Combined dataset", fill_color='#001126')
        # p.line(results['steps'][0], 1-results['train'].mean(0), legend_label="Combined dataset", line_width=3, line_color='#001126', line_dash='dashed')
        stdevshade(p, results['steps'][0], 1-results['test'], legend_label="Combined dataset", fill_color='#001126')
        p.line(results['steps'][0], 1-results['test'].mean(0), legend_label="Combined dataset", line_width=3, line_color='#001126')
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        save(p)

        # p = figure(border_fill_color='#f4f4f6', title="Linear vs nonlinear regression, D=1000", x_axis_label='Training steps', y_axis_label='R^2', height=300, y_range=(-.05,1.05))
        # output_file(f'{args.root}/plot12b.html')
        # p.add_layout(Legend(), 'left')
        # results = np.load(f'{args.root}/experiment10/results_regular.npz')
        # stdevshade(p, results['steps'][0], results['train'], legend_label="Original dataset", fill_color='#129490')
        # p.line(results['steps'][0], results['train'].mean(0), legend_label="Original dataset", line_width=3, line_color='#129490', line_dash='dashed')
        # stdevshade(p, results['steps'][0], results['test'], legend_label="Original dataset", fill_color='#129490')
        # p.line(results['steps'][0], results['test'].mean(0), legend_label="Original dataset", line_width=3, line_color='#129490')
        # results = np.load(f'{args.root}/experiment10/results_aux.npz')
        # stdevshade(p, results['steps'][0], results['train'], legend_label="Secondary dataset", fill_color='#90BE6D')
        # p.line(results['steps'][0], results['train'].mean(0), legend_label="Secondary dataset", line_width=3, line_color='#90BE6D', line_dash='dashed')
        # stdevshade(p, results['steps'][0], results['test'], legend_label="Secondary dataset", fill_color='#90BE6D')
        # p.line(results['steps'][0], results['test'].mean(0), legend_label="Secondary dataset", line_width=3, line_color='#90BE6D')
        # results = np.load(f'{args.root}/experiment10/results_regular_with_aux.npz')
        # stdevshade(p, results['steps'][0], results['train'], legend_label="Combined dataset", fill_color='#001126')
        # p.line(results['steps'][0], results['train'].mean(0), legend_label="Combined dataset", line_width=3, line_color='#001126', line_dash='dashed')
        # stdevshade(p, results['steps'][0], results['test'], legend_label="Combined dataset", fill_color='#001126')
        # p.line(results['steps'][0], results['test'].mean(0), legend_label="Combined dataset", line_width=3, line_color='#001126')
        # results = np.load(f'{args.root}/experiment10/results_finetune.npz')
        # stdevshade(p, results['steps'][0], results['train'], legend_label="Fine-tuned", fill_color='#175485')
        # p.line(results['steps'][0], results['train'].mean(0), legend_label="Fine-tuned", line_width=3, line_color='#175485', line_dash='dashed')
        # stdevshade(p, results['steps'][0], results['test'], legend_label="Fine-tuned", fill_color='#175485')
        # p.line(results['steps'][0], results['test'].mean(0), legend_label="Fine-tuned", line_width=3, line_color='#175485')
        # p.legend.click_policy = "hide"
        # save(p)
    except FileNotFoundError: print("File missing for experiment10, skipping...")

import argparse
import os
import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import Legend

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='./results', help="root dir to save results")
    args = parser.parse_args()

    ## Linear regression warm-up
    if os.path.exists(f'{args.root}/experiment01'):
        p = figure(title="Linear regression, D=1000", x_axis_label='Training steps', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment01/plot.html')
        p.add_layout(Legend(), 'right')
        results = np.load(f'{args.root}/experiment01/results.npz')
        p.line(results['steps'], results['train'], legend_label="In-sample", line_width=2, line_color='black', line_dash='dashed')
        p.line(results['steps'], results['test'], legend_label="Out-of-sample", line_width=2, line_color='black')
        p.legend.click_policy = "hide"
        save(p)

    ## Linear regression warm-up, but with L2 loss
    if os.path.exists(f'{args.root}/experiment02'):
        p = figure(title="Linear regression, D=1000", x_axis_label='Training steps', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment02/plot.html')
        p.add_layout(Legend(), 'right')
        results = np.load(f'{args.root}/experiment01/results.npz')
        p.line(results['steps'], results['train'], line_width=2, line_color='black', line_dash='dashed')
        p.line(results['steps'], results['test'], legend_label="Unregularized", line_width=2, line_color='black')
        colors = ['#308FFF', '#31B2E8', '#31E8C2']
        for i, wd in enumerate([.1, 1., 10.]):
            results = np.load(f'{args.root}/experiment02/results{wd:06.2f}.npz')
            p.line(results['steps'], results['train'], line_width=2, line_color=colors[i], line_dash='dashed')
            p.line(results['steps'], results['test'], legend_label=f"L2={wd}", line_width=2, line_color=colors[i])
        p.legend.click_policy = "hide"
        save(p)

    ## Linear regression on bigger data
    if os.path.exists(f'{args.root}/experiment03'):
        p = figure(title="Linear regression",  x_axis_label='Training steps', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment03/plot.html')
        p.add_layout(Legend(), 'right')
        small_data_results = np.load(f'{args.root}/experiment01/results.npz')
        p.line(small_data_results["steps"], small_data_results["train"], line_width=2, line_color='black', line_dash='dashed')
        p.line(small_data_results["steps"], small_data_results["test"], legend_label="D=1000", line_width=2, line_color='black')
        results = np.load(f'{args.root}/experiment03/results.npz')
        p.line(results["steps"], results["train"], line_width=2, line_color='blue', line_dash='dashed')
        p.line(results["steps"], results["test"], legend_label="D=10000", line_width=2, line_color='blue')
        p.legend.click_policy = "hide"
        save(p)

    ## Linear regression on various dataset sizes
    if os.path.exists(f'{args.root}/experiment04'):
        p = figure(title="Linear regression", x_axis_label='Dataset size', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment04/plot.html')
        p.add_layout(Legend(), 'right')
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, legend_label=f"In-sample", line_width=2, line_color='black', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Out-of-sample", line_width=2, line_color='black')
        p.legend.click_policy = "hide"
        save(p)

    ## Linear regression on various dataset sizes, augmenting input features with their squares
    if os.path.exists(f'{args.root}/experiment05'):
        p = figure(title="Linear regression", x_axis_label='Dataset size', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment05/plot.html')
        p.add_layout(Legend(), 'right')
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='black', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Regular features", line_width=2, line_color='black')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment05/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='blue', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Square features", line_width=2, line_color='blue')
        p.legend.click_policy = "hide"
        save(p)

    ## Linear regression on various dataset sizes, augmenting input features with all pairwise products
    if os.path.exists(f'{args.root}/experiment06'):
        p = figure(title="Linear regression", x_axis_label='Dataset size', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment06/plot.html')
        p.add_layout(Legend(), 'right')
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='black', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Regular features", line_width=2, line_color='black')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment05/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='blue', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Square features", line_width=2, line_color='blue')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment06/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='green', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Pair-prod features", line_width=2, line_color='green')
        p.legend.click_policy = "hide"
        save(p)

    ## Deep learning on small data
    if os.path.exists(f'{args.root}/experiment07'):
        p = figure(title="Linear vs nonlinear regression, D=1000", x_axis_label='Training steps', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment07/plot.html')
        p.add_layout(Legend(), 'right')
        results = np.load(f'{args.root}/experiment01/results.npz')
        p.line(results['steps'], results['train'], line_width=2, line_color='black', line_dash='dashed')
        p.line(results['steps'], results['test'], legend_label="Regular features", line_width=2, line_color='black')
        results = np.load(f'{args.root}/experiment07/results_lr.npz')
        p.line(results['steps'], results['train'], line_width=2, line_color='green', line_dash='dashed')
        p.line(results['steps'], results['test'], legend_label="Pair-prod features", line_width=2, line_color='green')
        results = np.load(f'{args.root}/experiment07/results_dl.npz')
        p.line(results['steps'], results['train'], line_width=2, line_color='purple', line_dash='dashed')
        p.line(results['steps'], results['test'], legend_label="Nonlinear regression", line_width=2, line_color='purple')
        p.legend.click_policy = "hide"
        save(p)

    ## Deep learning on various dataset sizes
    if os.path.exists(f'{args.root}/experiment08'):
        p = figure(title="Linear regression vs deep learning", x_axis_label='Dataset size', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment08/plot.html')
        p.add_layout(Legend(), 'right')
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment04/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='black', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Regular features", line_width=2, line_color='black')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment06/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='green', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Pair-prod features", line_width=2, line_color='green')
        r2_train = []; r2_test = []
        for d in dataset_sizes:
            results = np.load(f'{args.root}/experiment08/results{d:05}.npz')
            r2_train.append(results['train'][-1])
            r2_test.append(results['test'][-1])
        p.line(dataset_sizes, r2_train, line_width=2, line_color='purple', line_dash='dashed')
        p.line(dataset_sizes, r2_test, legend_label=f"Nonlinear regression", line_width=2, line_color='purple')
        p.legend.click_policy = "hide"
        save(p)


    ## Deep learning at various model sizes on various dataset sizes
    if os.path.exists(f'{args.root}/experiment09'):
        p = figure(title="Deep learning", x_axis_label='Dataset size', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment09/plot1.html')
        p.add_layout(Legend(), 'right')
        colors = ['#E138FF', '#A938EB', '#8B3DFA', '#5C38EB', '#3336FF', '#0A69FE']
        model_size_names = ['Small', 'Medium', 'Large']
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        for i, m in enumerate(model_size_names):
            r2_train = []; r2_test = []
            for d in dataset_sizes:
                results = np.load(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz')
                r2_train.append(results['train'][-1])
                r2_test.append(results['test'][-1])
            p.line(dataset_sizes, r2_train, line_width=2, line_color=colors[i], line_dash='dashed')
            p.line(dataset_sizes, r2_test, legend_label=m, line_width=2, line_color=colors[i])
        p.legend.click_policy = "hide"
        save(p)

        p = figure(title="Deep learning", x_axis_label='Dataset size', y_axis_label='R^2', height=300)
        output_file(f'{args.root}/experiment09/plot2.html')
        p.add_layout(Legend(), 'right')
        model_size_names = ['Small', 'Medium', 'Large', 'Huge', 'Enormous', 'XXL']
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        for i, m in enumerate(model_size_names):
            r2_train = []; r2_test = []
            for d in dataset_sizes:
                results = np.load(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz')
                r2_train.append(results['train'][-1])
                r2_test.append(results['test'][-1])
            # p.line(dataset_sizes, r2_train, line_width=1, line_color=colors[i], line_dash='dashed')
            p.line(dataset_sizes, r2_test, legend_label=m, line_width=1, line_color=colors[i])
        p.legend.click_policy = "hide"
        save(p)

        p = figure(title="Deep learning", x_axis_label='Model size (parameter count)', y_axis_label='R^2', height=300, x_axis_type='log')
        output_file(f'{args.root}/experiment09/plot3.html')
        p.add_layout(Legend(), 'right')
        model_shapes = [[32, 32, 16], [64, 64, 32], [256]*4, [512]*4, [1024]*5]
        def ms_to_pc(ms): return sum(il*ol+ol for il, ol in zip([256]+ms, ms+[1]))
        model_pcs = [ms_to_pc(ms) for ms in model_shapes]
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        for j, d in enumerate(dataset_sizes):
            r2_train = []; r2_test = []
            for i, m in enumerate(model_pcs):
                results = np.load(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz')
                r2_train.append(results['train'][-1])
                r2_test.append(results['test'][-1])
            # p.line(model_pcs, r2_train, line_width=1, line_color=colors[j], line_dash='dashed')
            p.line(model_pcs, r2_test, legend_label=f'D={d}', line_width=1, line_color=colors[j])
        p.legend.click_policy = "hide"
        save(p)

        p = figure(title="Deep learning", x_axis_label='Model size (parameter count)', y_axis_label='R^2', height=300, x_axis_type='log')
        output_file(f'{args.root}/experiment09/plot4.html')
        p.add_layout(Legend(), 'right')
        model_shapes = [[32, 32, 16], [64, 64, 32], [256]*4, [512]*4, [1024]*5, [2048]*5]
        def ms_to_pc(ms): return sum(il*ol+ol for il, ol in zip([256]+ms, ms+[1]))
        model_pcs = [ms_to_pc(ms) for ms in model_shapes]
        dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        for j, d in enumerate(dataset_sizes):
            r2_train = []; r2_test = []
            for i, m in enumerate(model_pcs):
                results = np.load(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz')
                r2_train.append(results['train'][-1])
                r2_test.append(results['test'][-1])
            # p.line(model_pcs, r2_train, line_width=1, line_color=colors[j], line_dash='dashed')
            p.line(model_pcs, r2_test, legend_label=f'D={d}', line_width=1, line_color=colors[j])
        p.legend.click_policy = "hide"
        save(p)

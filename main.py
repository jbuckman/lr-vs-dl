import os
import random

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Subset, IterableDataset, ConcatDataset, Dataset
from torchvision.datasets import EMNIST
from functools import partial
import argparse
import pandas as pd
from PIL import Image, ImageOps

class InputTransform:
    def __call__(self, sample):
        out = torch.tensor(np.array(ImageOps.mirror(sample.rotate(-90)))[6:22,6:22].reshape(-1).astype(float)/128. - 1., dtype=torch.float32)
        out[0:5] = 2
        return out
class InputTransform2:
    def __call__(self, sample):
        out = torch.tensor(np.array(sample)[6:22,6:22].reshape(-1).astype(float)/128. - 1., dtype=torch.float32)
        out[0:5] = -2
        return out

class InputTransformAugmentSquares(InputTransform):
    def __call__(self, sample):
        base = super().__call__(sample)
        return torch.cat([base, base**2], 0)

class InputTransformAugmentPairwiseProducts(InputTransform):
    def __call__(self, sample):
        base = super().__call__(sample)
        aug = base[:,None]
        aug = (aug * aug.T)[np.triu_indices(base.shape[0])]
        return torch.cat([base, aug], 0)

class TargetTransform:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)
class TargetTransform2:
    def __call__(self, sample):
        return torch.tensor(sample + 10, dtype=torch.float32)

def make_dataset_main(train, size, transform=InputTransform):
    dataset = EMNIST('/tmp/datasets', split="mnist", train=train, download=True, transform=transform(), target_transform=TargetTransform())
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    return Subset(dataset, idx[:size])

class TMNISTDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.data = pd.read_csv("TMNIST_Data.csv")
        self.data = self.data[[c for c in self.data.columns if c != 'names']].to_numpy().astype(int)
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.data[idx][1:].reshape(28,28)
        target = int(self.data[idx][0])
        if self.transform: image = self.transform(image)
        if self.target_transform: target = self.target_transform(target)
        return image, target

def make_dataset_aux(train, size):
    if train:
        aux = TMNISTDataset(transform=InputTransform2(), target_transform=TargetTransform())
        idx = np.arange(len(aux))
        np.random.shuffle(idx)
        aux = Subset(aux, idx[:25_000])
        return aux
    else:
        return make_dataset_main(False, size)

def make_dataset_main_plus_aux(train, size):
    if train:
        main = make_dataset_main(True, size, InputTransform)
        aux = TMNISTDataset(transform=InputTransform2(), target_transform=TargetTransform())
        idx = np.arange(len(aux))
        np.random.shuffle(idx)
        aux = Subset(aux, idx[:25_000])
        return ConcatDataset([main, aux])
    else:
        return make_dataset_main(False, size)

class InfiniteData(IterableDataset):
    def __init__(self, ds):
        self.ds = ds
    def __iter__(self):
        return self
    def __next__(self):
        return self.ds[np.random.randint(0, len(self.ds))]

class FiniteData(IterableDataset):
    def __init__(self, ds, amount=None):
        self.ds = ds
        self.amount = min(len(ds), amount) if amount is not None else len(ds)
    def __len__(self):
        return self.amount
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i == self.amount: raise StopIteration
        out = self.ds[self.i]
        self.i += 1
        return out

class LinearModel(nn.Module):
    def __init__(self, input_size=256):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, 1)
        self.reset_parameters()

    def forward(self, x):
        return self.l1(x).squeeze()

    def reset_parameters(self, initial_params=None):
        if initial_params is None:
            nn.init.kaiming_normal_(self.l1.weight, a=1, mode='fan_in')
        else:
            self.load_state_dict(initial_params)

class FeedforwardNet(torch.nn.Module):
    def __init__(self, input_size, layers, nonlin=nn.ReLU(inplace=True)):
        super().__init__()
        self.nonlin = nonlin
        layer_widths = [input_size] + layers + [1]
        self.hidden_layers = torch.nn.ModuleList([
            nn.Linear(in_d, out_d)
            for in_d, out_d in zip(layer_widths[:-2], layer_widths[1:-1])])
        self.final_layer = nn.Linear(layer_widths[-2], layer_widths[-1])
        self.reset_parameters()

    def reset_parameters(self, initial_params=None):
        if initial_params is None:
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight, a=1, mode='fan_in')
            nn.init.zeros_(self.final_layer.weight)
        else:
            self.load_state_dict(initial_params)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        for i, layer in enumerate(self.hidden_layers):
            x = self.nonlin(layer(x))
        x = self.final_layer(x)
        return x.squeeze()

def training_step(model, batch, opt):
    xs, ys = batch
    if torch.cuda.is_available(): xs = xs.to('cuda', non_blocking=True); ys = ys.to('cuda', non_blocking=True)
    preds = model(xs)
    losses = (preds - ys)**2.
    loss = losses.mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

def evaluate(model, dataset, ss_tot):
    ss_res = 0
    for batch in dataset:
        xs, ys = batch
        if torch.cuda.is_available(): xs = xs.to('cuda', non_blocking=True); ys = ys.to('cuda', non_blocking=True)
        preds = model(xs).round()
        ss_res += ((preds - ys)**2.).sum()
    return (1 - ss_res/ss_tot).item()

def experiment(model, trials=3, make_dataset=make_dataset_main, lr=1e-3, wd=0., batch_size=256, total_steps=100000,
               train_set_size=1000, eval_set_size=10000, eval_intermediate=False, return_params=False, initial_params=None):
    if torch.cuda.is_available(): model.to('cuda')

    all_results_steps = []
    all_results_train = []
    all_results_test = []
    for trial in range(trials):
        print(f"Trial {trial+1}")
        train_ds = make_dataset(train=True, size=train_set_size)
        eval_train_ds = FiniteData(train_ds, eval_set_size)
        test_ds = make_dataset(train=False, size=eval_set_size)

        train_loader = DataLoader(InfiniteData(train_ds), batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
        eval_train_loader = DataLoader(eval_train_ds, drop_last=False, batch_size=batch_size*2, pin_memory=torch.cuda.is_available())
        eval_test_loader = DataLoader(FiniteData(test_ds, eval_set_size), batch_size=batch_size*2, drop_last=False, pin_memory=torch.cuda.is_available())

        train_mean = sum([t for _, t in eval_train_ds])/len(eval_train_ds)
        train_ss = sum([(t - train_mean)**2 for _, t in eval_train_ds])
        test_mean = sum([t for _, t in test_ds])/len(test_ds)
        test_ss = sum([(t - test_mean)**2 for _, t in test_ds])

        model.reset_parameters(initial_params)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        step = 0
        results_steps = []
        results_train = []
        results_test = []
        if eval_intermediate:
            results_steps.append(step)
            results_train.append(evaluate(model, eval_train_loader, train_ss))
            results_test.append(evaluate(model, eval_test_loader, test_ss))
            print(f"Step {step: 8}   |   In-sample R^2 {results_train[-1]:.2f}   |   Out-of-sample R^2 {results_test[-1]:.2f}", flush=True)
        for batch in train_loader:
            training_step(model, batch, opt)
            step += 1
            if step % (total_steps // 10) == 0:
                print(f"{step}/{total_steps} ({step / total_steps:.0%})")
            if step == total_steps or (eval_intermediate and np.log2(step) % 1 == 0):
                results_steps.append(step)
                results_train.append(evaluate(model, eval_train_loader, train_ss))
                results_test.append(evaluate(model, eval_test_loader, test_ss))
                print(f"Step {step: 8}   |   In-sample R^2 {results_train[-1]:.2f}   |   Out-of-sample R^2 {results_test[-1]:.2f}", flush=True)
            if step == total_steps:
                break
        all_results_steps.append(results_steps)
        all_results_train.append(results_train)
        all_results_test.append(results_test)
    if not return_params:
        return all_results_steps, all_results_train, all_results_test
    else:
        return all_results_steps, all_results_train, all_results_test, model.state_dict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=int, default=1, help="choose which experiment to run")
    parser.add_argument("--root", default='./results', help="root dir to save results")
    args = parser.parse_args()

    ## Linear regression warm-up
    if args.experiment == 1:
        os.makedirs(f'{args.root}/experiment01', exist_ok=True)
        results_steps, results_train, results_test = experiment(LinearModel(), total_steps=1_000_000, train_set_size=1000, eval_intermediate=True)
        np.savez(f'{args.root}/experiment01/results.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression warm-up, but with L2 loss
    elif args.experiment == 2:
        os.makedirs(f'{args.root}/experiment02', exist_ok=True)
        for wd in [.05, .1, .2]:
            print(f"Running {wd}")
            results_steps, results_train, results_test = experiment(LinearModel(), wd=wd, total_steps=1_000_000, train_set_size=1000, eval_intermediate=True)
            np.savez(f'{args.root}/experiment02/results{wd:06.2f}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on bigger data
    elif args.experiment == 3:
        os.makedirs(f'{args.root}/experiment03', exist_ok=True)
        results_steps, results_train, results_test = experiment(LinearModel(), total_steps=1_000_000, train_set_size=10000, eval_intermediate=True)
        np.savez(f'{args.root}/experiment03/results.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on various dataset sizes
    elif args.experiment == 4:
        os.makedirs(f'{args.root}/experiment04', exist_ok=True)
        for d in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
            print(f"Running {d}")
            results_steps, results_train, results_test = experiment(LinearModel(), total_steps=1_000_000, train_set_size=d, eval_intermediate=False)
            np.savez(f'{args.root}/experiment04/results{d:05}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on various dataset sizes, augmenting input features with their squares
    elif args.experiment == 5:
        os.makedirs(f'{args.root}/experiment05', exist_ok=True)
        for d in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
            print(f"Running {d}")
            results_steps, results_train, results_test = experiment(LinearModel(256*2), total_steps=1_000_000, train_set_size=d, eval_intermediate=False,
                                                                    make_dataset=partial(make_dataset_main, transform=InputTransformAugmentSquares))
            np.savez(f'{args.root}/experiment05/results{d:05}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on various dataset sizes, augmenting input features with all pairwise products
    elif args.experiment == 6:
        os.makedirs(f'{args.root}/experiment06', exist_ok=True)
        for d in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
            print(f"Running {d}")
            results_steps, results_train, results_test = experiment(LinearModel(256 + 257*256//2), total_steps=2000, train_set_size=d, eval_intermediate=False,
                                                                    make_dataset=partial(make_dataset_main, transform=InputTransformAugmentPairwiseProducts))
            np.savez(f'{args.root}/experiment06/results{d:05}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Deep learning on small data
    elif args.experiment == 7:
        os.makedirs(f'{args.root}/experiment07', exist_ok=True)
        results_steps, results_train, results_test = experiment(LinearModel(256), total_steps=1_000_000, train_set_size=1000, eval_intermediate=True)
        np.savez(f'{args.root}/experiment07/results_linear.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))
        results_steps, results_train, results_test = experiment(FeedforwardNet(256, [32, 32, 16]), total_steps=1_000_000, train_set_size=1000, eval_intermediate=True)
        np.savez(f'{args.root}/experiment07/results_dl.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Deep learning on various dataset sizes
    elif args.experiment == 8:
        os.makedirs(f'{args.root}/experiment08', exist_ok=True)
        for d in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
            print(f"Running {d}")
            results_steps, results_train, results_test = experiment(FeedforwardNet(256, [32, 32, 16]),
                                                                    total_steps=1_000_000, train_set_size=d,
                                                                    eval_intermediate=False)
            np.savez(f'{args.root}/experiment08/results{d:05}.npz', steps=np.array(results_steps),
                     train=np.array(results_train), test=np.array(results_test))

    ## Deep learning on various model/dataset sizes
    elif args.experiment == 9:
        os.makedirs(f'{args.root}/experiment09', exist_ok=True)
        for i, m in enumerate([[32, 32, 16], [64, 64, 32], [128, 128, 64, 64], [256] * 4, [512] * 4]):
            for d in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
                print(f"Running {m} {d}")
                results_steps, results_train, results_test = experiment(FeedforwardNet(256, m), total_steps=1_000_000, train_set_size=d, eval_intermediate=False, trials=3)
                np.savez(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Deep learning on small data, plus an auxilliary task
    elif args.experiment == 10:
        os.makedirs(f'{args.root}/experiment10', exist_ok=True)
        results_steps, results_train, results_test = experiment(FeedforwardNet(256, [256]*4), total_steps=1_000_000, train_set_size=250, eval_set_size=None, eval_intermediate=True, make_dataset=make_dataset_main)
        np.savez(f'{args.root}/experiment10/results_regular.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))
        results_steps, results_train, results_test, init_for_finetune = experiment(FeedforwardNet(256, [256]*4), total_steps=1_000_000, train_set_size=250, eval_set_size=None, eval_intermediate=True, make_dataset=make_dataset_aux, return_params=True)
        np.savez(f'{args.root}/experiment10/results_aux.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))
        results_steps, results_train, results_test = experiment(FeedforwardNet(256, [256]*4), total_steps=1_000_000, train_set_size=250, eval_set_size=None, eval_intermediate=True, make_dataset=make_dataset_main_plus_aux)
        np.savez(f'{args.root}/experiment10/results_regular_with_aux.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))
        results_steps, results_train, results_test = experiment(FeedforwardNet(256, [256]*4), initial_params=init_for_finetune, total_steps=1_000_000, train_set_size=250, eval_set_size=None, eval_intermediate=True, make_dataset=make_dataset_main)
        np.savez(f'{args.root}/experiment10/results_finetune.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    else:
        raise Exception(f"No experiment with ID {args.experiment}")

    ## for i in 1 2 3 4 5 6 7 8 9 10; do echo "experiment $i"; python main.py $i; done
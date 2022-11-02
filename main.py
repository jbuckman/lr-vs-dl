import os

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Subset, IterableDataset, ConcatDataset
from torchvision.datasets import EMNIST
from functools import partial
import argparse

class InputTransform:
    def __call__(self, sample):
        return torch.tensor(np.array(sample)[6:22,6:22].reshape(-1).astype(float)/128. - 1., dtype=torch.float32)

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

def make_dataset_main(train, size, transform=InputTransform):
    return Subset(EMNIST('/tmp/datasets', split="mnist", train=train, download=True, transform=transform(), target_transform=TargetTransform()), range(size))

def make_dataset_both(train, size, transform=InputTransform):
    if train:
        aux = Subset(EMNIST('/tmp/datasets', split="letters", train=train, download=True, transform=transform(), target_transform=TargetTransform()), range(100_000))
        return ConcatDataset(make_dataset_main(True, size_main, transform), aux)
    else:
        return make_dataset_main(False, size)

class InfiniteData(IterableDataset):
    def __init__(self, ds):
        self.ds = ds
    def __iter__(self):
        yield self.ds[np.random.randint(0, len(self.ds))]

class LinearModel(nn.Module):
    def __init__(self, input_size=256):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.l1(x).squeeze()

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

    def reset_parameters(self):
        for layer in self.hidden_layers:
            nn.init.kaiming_normal_(layer.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.final_layer.weight)

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
        preds = model(xs)
        ss_res += ((preds - ys)**2.).sum()
    return (1 - ss_res/ss_tot).item()

def experiment(model, make_dataset=make_dataset_main, lr=3e-4, wd=0., batch_size=256, total_steps=100000,
               train_set_size=1000, eval_set_size=1000, eval_every=None, eval_train=True):
    if torch.cuda.is_available(): model.to('cuda')

    train_ds = make_dataset(train=True, size=train_set_size)
    if eval_train: eval_train_ds = make_dataset(train=True, size=min(eval_set_size, train_set_size))
    eval_test_ds = make_dataset(train=False, size=eval_set_size)

    train_loader = DataLoader(InfiniteData(train_ds), batch_size=batch_size, drop_last=False, pin_memory=torch.cuda.is_available())
    if eval_train: eval_train_loader = DataLoader(eval_train_ds, batch_size=batch_size*2, drop_last=False, pin_memory=torch.cuda.is_available())
    eval_test_loader = DataLoader(eval_test_ds, batch_size=batch_size*2, drop_last=False, pin_memory=torch.cuda.is_available())

    if eval_train:
        train_mean = sum([t for _, t in eval_train_ds])/len(eval_train_ds)
        train_ss = sum([(t - train_mean)**2 for _, t in eval_train_ds])
    test_mean = sum([t for _, t in eval_test_ds])/len(eval_test_ds)
    test_ss = sum([(t - test_mean)**2 for _, t in eval_test_ds])

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    step = 0
    results_steps = []
    if eval_train: results_train = []
    results_test = []
    while step < total_steps:
        for batch in train_loader:
            training_step(model, batch, opt)
            step += 1
            if step % (total_steps // 100) == 0:
                print(f"{step}/{total_steps} ({step / total_steps:.0%})")
            if step == total_steps or (eval_every is not None and step % eval_every == 0):
                results_steps.append(step)
                if eval_train: results_train.append(evaluate(model, eval_train_loader, train_ss))
                results_test.append(evaluate(model, eval_test_loader, test_ss))
                print(f"Step {step: 8}   |   In-sample R^2 {results_train[-1] if eval_train else 0.:.2f}   |   Out-of-sample R^2 {results_test[-1]:.2f}", flush=True)
            if step == total_steps:
                break
    if eval_train:
        return results_steps, results_train, results_test
    else:
        return results_steps, results_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=int, default=1, help="choose which experiment to run")
    parser.add_argument("--root", default='./results', help="root dir to save results")
    args = parser.parse_args()

    ## Linear regression warm-up
    if args.experiment == 1:
        os.makedirs(f'{args.root}/experiment01', exist_ok=True)
        results_steps, results_train, results_test = experiment(LinearModel(), total_steps=200000, train_set_size=1000, eval_every=5000)
        np.savez(f'{args.root}/experiment01/results.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression warm-up, but with L2 loss
    elif args.experiment == 2:
        os.makedirs(f'{args.root}/experiment02', exist_ok=True)
        for wd in [.01, .1, 1., 10., 100.]:
            results_steps, results_train, results_test = experiment(LinearModel(), wd=wd, total_steps=200000, train_set_size=1000, eval_every=5000)
            np.savez(f'{args.root}/experiment02/results{wd:06.2f}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on bigger data
    elif args.experiment == 3:
        os.makedirs(f'{args.root}/experiment03', exist_ok=True)
        results_steps, results_train, results_test = experiment(LinearModel(), total_steps=200000, train_set_size=10000, eval_every=5000)
        np.savez(f'{args.root}/experiment03/results.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on various dataset sizes
    elif args.experiment == 4:
        os.makedirs(f'{args.root}/experiment04', exist_ok=True)
        for d in [500, 1000, 2000, 5000, 10000, 20000]:
            results_steps, results_train, results_test = experiment(LinearModel(), total_steps=200000, train_set_size=d, eval_set_size=10000, eval_every=None)
            np.savez(f'{args.root}/experiment04/results{d:05}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on various dataset sizes, augmenting input features with their squares
    elif args.experiment == 5:
        os.makedirs(f'{args.root}/experiment05', exist_ok=True)
        for d in [500, 1000, 2000, 5000, 10000, 20000]:
            results_steps, results_train, results_test = experiment(LinearModel(256*2), total_steps=200000, train_set_size=d, eval_set_size=10000, eval_every=None,
                                                                    make_dataset=partial(make_dataset_main, transform=InputTransformAugmentSquares))
            np.savez(f'{args.root}/experiment05/results{d:05}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Linear regression on various dataset sizes, augmenting input features with all pairwise products
    elif args.experiment == 6:
        os.makedirs(f'{args.root}/experiment06', exist_ok=True)
        for d in [500, 1000, 2000, 5000, 10000, 20000]:
            results_steps, results_train, results_test = experiment(LinearModel(256 + 257*256//2), total_steps=200000, train_set_size=d, eval_set_size=10000, eval_every=None,
                                                                    make_dataset=partial(make_dataset_main, transform=InputTransformAugmentPairwiseProducts), lr=1e-5)
            np.savez(f'{args.root}/experiment06/results{d:05}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Deep learning on small data
    if args.experiment == 7:
        os.makedirs(f'{args.root}/experiment07', exist_ok=True)
        results_steps, results_train, results_test = experiment(LinearModel(256 + 257*256//2), total_steps=200000, train_set_size=1000, eval_every=5000,
                                                                make_dataset=partial(make_dataset_main, transform=InputTransformAugmentPairwiseProducts), lr=1e-5)
        np.savez(f'{args.root}/experiment07/results_lr.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))
        results_steps, results_train, results_test = experiment(FeedforwardNet(256, [32, 32, 16]), total_steps=200000, train_set_size=1000, eval_every=5000)
        np.savez(f'{args.root}/experiment07/results_dl.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Deep learning on various dataset sizes
    elif args.experiment == 8:
        os.makedirs(f'{args.root}/experiment08', exist_ok=True)
        for d in [500, 1000, 2000, 5000, 10000, 20000]:
            results_steps, results_train, results_test = experiment(FeedforwardNet(256, [32, 32, 16]), total_steps=200000, train_set_size=d, eval_set_size=10000, eval_every=None)
            np.savez(f'{args.root}/experiment08/results{d:05}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))

    ## Deep learning on various model/dataset sizes
    elif args.experiment == 9:
        os.makedirs(f'{args.root}/experiment09', exist_ok=True)
        for i, m in enumerate([[32, 32, 16], [64, 64, 32], [256]*4, [512]*4, [1024]*5, [2048]*5]):
            for d in [500, 1000, 2000, 5000, 10000, 20000]:
                results_steps, results_train, results_test = experiment(FeedforwardNet(256, m), total_steps=500000, train_set_size=d, eval_set_size=10000, eval_every=None)
                np.savez(f'{args.root}/experiment09/results_d{d:05}_m{i}.npz', steps=np.array(results_steps), train=np.array(results_train), test=np.array(results_test))


    else:
        raise Exception(f"No experiment with ID {args.experiment}")
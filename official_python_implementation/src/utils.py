import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

import pandas as pd

from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from src import ff_mnist, ff_model
import wandb


def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.classification_loss.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.classification_loss.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer
# 784, 2000, 2000, 2000 # main params
# 6000, 10 # classification_loss params

def get_data(opt, partition):
    # dataset = ff_mnist.FF_MNIST(opt, partition)
    dataset = ff_mnist.FF_senti(opt, partition, num_classes=2)
    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=1,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_senti_partition(opt, partition):
    # load reviews data
    # print(os.path.join(get_original_cwd(), opt.input.training_path))
    train_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.training_path), names=["filename", "split", "labels", "features"])
    test_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.test_path), names=["filename", "split", "labels", "features"])
    # train_df = pd.read_csv('reviews_train.csv',names=["filename", "split", "labels", "features"])
    # test_df = pd.read_csv('reviews_test.csv',names=["filename", "split", "labels", "features"])
    train_df = train_df.drop(columns=["filename", "split"])
    test_df = test_df.drop(columns=["filename", "split"])
    train_df['labels'] = train_df['labels'].replace({'pos': 1, 'neg': 0})
    test_df['labels'] = test_df['labels'].replace({'pos': 1, 'neg': 0})

    train_labels = torch.tensor(train_df['labels'].values, dtype=torch.long)
    test_labels = torch.tensor(test_df['labels'].values, dtype=torch.long)

    train_data = train_df['features'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32))
    train_data = torch.stack([torch.tensor(x) for x in train_data])

    test_data = test_df['features'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32))
    test_data = torch.stack([torch.tensor(x) for x in test_data])

    final_train_data = torch.hstack((train_data,torch.unsqueeze(train_labels, 1)))
    final_test_data = torch.hstack((test_data,torch.unsqueeze(test_labels, 1)))

    if partition in ["train"]:
        return train_data, train_labels
    else:
        return test_data, test_labels

def get_MNIST_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # Lambda(lambda x: torch.flatten(x)),
        ]
    )
    if partition in ["train"]:
        mnist = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        mnist = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    # if partition == "train":
    #     mnist = torch.utils.data.Subset(mnist, range(40000))
    # elif partition == "val":
    #     mnist = torchvision.datasets.CIFAR10(
    #         os.path.join(get_original_cwd(), opt.input.path),
    #         train=True,
    #         download=True,
    #         transform=transform,
    #     )
    #     mnist = torch.utils.data.Subset(mnist, range(40000, 50000))

    return mnist


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels

# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)

# create save_model function
def save_model(model):
    torch.save(model.state_dict(), f"{wandb.run.name}-model.pt")
    # log model to wandb
    wandb.save(f"{wandb.run.name}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict

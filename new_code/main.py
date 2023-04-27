import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig

from src import utils
import wandb
from torch.nn import ReLU, Tanh, GELU, Sigmoid, SELU, ELU, CELU

def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)
    best_val_acc = 0.0

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels) # push to GPU

            # print("input shape:",inputs['sample'].shape)
            # print("label shape:",labels['class_labels'].shape)
            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            best_val_acc = validate_or_test(opt, model, "val", epoch=epoch, best_val_acc=best_val_acc)
            # utils.print_results("val", time.time() - start_time, train_results, epoch)

    return model


def validate_or_test(opt, model, partition, epoch=None, best_val_acc=1.0):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            scalar_outputs = model.forward_downstream_multi_pass(
                inputs, labels, scalar_outputs=scalar_outputs
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    # save model if classification accuracy is better than previous best
    if test_results["classification_accuracy"] > best_val_acc:
        print("saving model")
        best_val_acc = test_results["classification_accuracy"]
        utils.save_model(model)

    model.train()
    return best_val_acc


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    run = wandb.init(
    project="project",
    entity  = "automellon",
    name = "mnist 2000 threshhold", # Wandb creates random run names if you skip this field
    reinit = False, # Allows reinitalizing runs when you re-run this cell
    # run_id = # Insert specific run id here if you want to resume a previous run
    # resume = "must" # You need this to resume previous runs, but comment out reinit = True when using this
    config = dict(opt) ### Wandb Config for your run
    )
    activations = [ReLU(), Tanh(), GELU(), Sigmoid(), SELU(), ELU(), CELU()]
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    run.finish()
    # validate_or_test(opt, model, "val")

    # if opt.training.final_test:
    #     validate_or_test(opt, model, "test")


if __name__ == "__main__":
    my_main()

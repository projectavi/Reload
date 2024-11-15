import copy
import os
from collections import OrderedDict

import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
import transform_dataset


def save_kv(data_loaders, model, criterion, args, run_id=""):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    old_gradients = {}
    old_loader = data_loaders["full"]

    new_gradients = {}
    new_loader = data_loaders["retain"]

    for name, param in model.named_parameters():
        old_gradients[name] = 0
        new_gradients[name] = 0

    for i, (image, target) in enumerate(old_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    old_gradients[name] += param.grad.data


    for i, (image, target) in enumerate(new_loader):
        image = image.to(device)
        target = target.to(device)

        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    new_gradients[name] += param.grad.data

    epsilon = 1e-10

    gradients = {}
    return_gradients = {}

    with torch.no_grad():
        for name in old_gradients:
            return_gradients[name] = old_gradients[name] - new_gradients[name]
            gradients[name] = ((torch.abs_(return_gradients[name]) + epsilon) / (torch.abs_(old_gradients[name]) + epsilon))

    # This gives the portion of the top k of the gradients that relate to the forget data to set to 1 in the mask
    if args.threshold is None:
        threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        threshold_list = [1 - args.threshold]

    for i in threshold_list:
        print(i)
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top i*100% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.ones_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 0
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        if run_id is None:
            if args.run_id is None:
                run_id = "test"
            else:
                run_id = args.run_id

        torch.save(hard_dict, os.path.join(args.save_dir, run_id + " with_{" + str(i if args.threshold is None else args.threshold) + "}.pt"))
    
    return return_gradients

def main(run_id, args=None):
    if args is None:
        args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.to(device)
    # print(model.state_dict())

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )

    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader, full=train_loader_full
    )
    
    trained_model = None

    if args.task == "replace" and args.replacement_type == "plausibly_incorrect_label":
        checkpoint = torch.load("./trained_models/0cifar100_resnet18_model_SA_best.pth.tar", map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        trained_model = copy.deepcopy(model).to(device)

        trained_model.load_state_dict(checkpoint, strict=False)

    if args.task == "replace":
        unlearn_data_loaders = utils.replace_data(unlearn_data_loaders, args, model=trained_model)
        retain_dataset = unlearn_data_loaders["retain"].dataset
        forget_dataset = unlearn_data_loaders["forget"].dataset

    del trained_model

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")

    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.mask, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]

    if args.unlearn != "retrain":
        model.load_state_dict(checkpoint, strict=False)

    gradients = save_kv(unlearn_data_loaders, model, criterion, args, run_id)

    return unlearn_data_loaders, gradients


if __name__ == "__main__":
    main("")
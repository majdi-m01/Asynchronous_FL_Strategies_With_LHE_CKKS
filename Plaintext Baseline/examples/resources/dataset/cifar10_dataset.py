import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    Dataset,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
    fixed_partition, hetero_dir_partition
)


def get_cifar10(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    """
    Return the CIFAR10 dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.CIFAR10(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # Training data for multiple clients
    train_data_raw = torchvision.datasets.CIFAR10(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    fixed_partition_table = {
        0: {0: 211, 1: 53, 2: 53, 3: 53, 4: 53, 5: 211, 6: 52, 7: 52, 8: 211, 9: 51},  # 1000
        1: {0: 51, 1: 643, 2: 213, 3: 51, 4: 213, 5: 51, 6: 213, 7: 51, 8: 213, 9: 51},  # 1750
        2: {0: 462, 1: 707, 2: 235, 3: 47, 4: 236, 5: 236, 6: 47, 7: 47, 8: 236, 9: 47},  # 2300
        3: {0: 424, 1: 223, 2: 223, 3: 223, 4: 942, 5: 47, 6: 47, 7: 223, 8: 223, 9: 225},  # 2800
        4: {0: 362, 1: 182, 2: 45, 3: 542, 4: 182, 5: 362, 6: 187, 7: 937, 8: 362, 9: 539},  # 3700
        5: {0: 400, 1: 400, 2: 400, 3: 400, 4: 400, 5: 766, 6: 192, 7: 575, 8: 575, 9: 192},  # 4300
        6: {0: 157, 1: 314, 2: 784, 3: 940, 4: 470, 5: 756, 6: 945, 7: 567, 8: 378, 9: 189},  # 5500
        7: {0: 312, 1: 939, 2: 965, 3: 312, 4: 312, 5: 1157, 6: 386, 7: 386, 8: 771, 9: 1160},  # 6700
        8: {0: 670, 1: 669, 2: 1172, 3: 168, 4: 670, 5: 769, 6: 1345, 7: 192, 8: 385, 9: 960},  # 7000
        9: {0: 1088, 1: 48, 2: 191, 3: 1338, 4: 765, 5: 382, 6: 1147, 7: 1338, 8: 956, 9: 1147}  # 8400
    }

    # Partition the dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_nomiid":
        train_datasets = dirichlet_noniid_partition(
            train_data_raw, num_clients, **kwargs  # TODO adjusted the alpha parameters
        )
    elif partition_strategy == "fixed":
        train_datasets = fixed_partition(
            train_data_raw,
            num_clients,
            fixed_partition_table,
            **kwargs,
        )
    elif partition_strategy == "hetero-dir":
        train_datasets = hetero_dir_partition(
            train_data_raw,
            num_clients,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    return train_datasets[client_id], test_dataset

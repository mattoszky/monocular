
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
import argparse

from utils.dataset import CustomDataset
from solvers.solver_denorm import Solver_Denorm

def main():
    # sets the seed
    seed = 254
    torch.manual_seed(seed)

    # takes argument from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--mn', type=str, required=True, help="Model Name")
    args = parser.parse_args()
    
    # loads the data
    file_path = "./../data/dataset/data20.dat"
    data = np.loadtxt(file_path)

    # gets inputs and GTs
    inputs = data[:, :5]
    gt = data[:, 5:]

    # creates the dataset
    dataset = CustomDataset(inputs, gt)
    
    # train-validation-test sizes
    total_size = len(dataset)
    train_size = int(0.7 * total_size)              # 70% for training
    val_size = int(0.2 * total_size)                # 20% for validation
    test_size = total_size - train_size - val_size  # 10% for test

    # creates the actual datasets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # gets the dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    max_epoch = 300
    lr = 0.0001
    inc_val = 30
    print_every = 10
    th_x = 10
    th_y = 5
    test_criterion = nn.L1Loss()
    train_criterion = nn.MSELoss()
    
    solver = Solver_Denorm(seed = seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=max_epoch,
                lr=lr,
                train_criterion=train_criterion,
                test_criterion=test_criterion,
                inc_val=inc_val,
                batch_size=batch_size,
                checkpoint_path="../models",
                model_name=args.mn, 
                print_every=print_every,
                th_x=th_x,
                th_y=th_y
            )
    solver.train()
    solver.test()

if __name__ == "__main__":
    main()
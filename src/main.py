
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import argparse
import time

from utils.dataset import CustomDataset
from solvers.solver import Solver

def main():
    seed = 199
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
    
    # uncomment for class + TL
    # inputs = inputs[:, :3]
    
    # uncomment for class + HW
    #inputs = inputs[:, [0, -2, -1]]
    
    # train-validation-test sizes
    total_size = len(inputs)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.2 * total_size)    # 20% for validation

    # indexes to divide data
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # applies the previous indexes to get the actual data
    train_inputs = inputs[train_indices]
    train_gt = gt[train_indices]
    val_inputs = inputs[val_indices]
    val_gt = gt[val_indices]
    test_inputs = inputs[test_indices]
    test_gt = gt[test_indices]

    # gets the minimum and the maximum of inputs and GTs of training
    min_vals_input = train_inputs.min(axis=0)
    max_vals_input = train_inputs.max(axis=0)
    min_vals_gt = train_gt.min(axis=0)
    max_vals_gt = train_gt.max(axis=0)

    # normalizes the data
    train_inputs = (train_inputs - min_vals_input) / (max_vals_input - min_vals_input)
    val_inputs = (val_inputs - min_vals_input) / (max_vals_input - min_vals_input)
    test_inputs = (test_inputs - min_vals_input) / (max_vals_input - min_vals_input)
    train_gt = (train_gt - min_vals_gt) / (max_vals_gt - min_vals_gt)
    val_gt = (val_gt - min_vals_gt) / (max_vals_gt - min_vals_gt)
    test_gt = (test_gt - min_vals_gt) / (max_vals_gt - min_vals_gt)
    
    # converts max and min in tensor 
    max_val_tensor = torch.cat((torch.from_numpy(max_vals_input), torch.from_numpy(max_vals_gt)), dim=0)
    min_val_tensor = torch.cat((torch.from_numpy(min_vals_input), torch.from_numpy(min_vals_gt)), dim=0)

    # gets the datasets
    train_dataset = CustomDataset(train_inputs, train_gt)
    val_dataset = CustomDataset(val_inputs, val_gt)
    test_dataset = CustomDataset(test_inputs, test_gt)

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
    #train_criterion = nn.SmoothL1Loss()
    #train_criterion = nn.HuberLoss(delta=0.8)
    #train_criterion = nn.HuberLoss(delta=0.3)
    
    solver = Solver(seed = seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                min_vals=min_val_tensor,
                max_vals=max_val_tensor,
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
    #solver.load_model() #if you want to load an existing model
    solver.train()
    solver.test()
    
    # here an example to use the trained model to get a result
    '''
    inp = torch.tensor([0.000000, 2500.000000, 839.000000, 117.000000, 167.000000], dtype=torch.float32)
    min_val_tensor = min_val_tensor.type(torch.float32)
    max_val_tensor = max_val_tensor.type(torch.float32)
    inp = (inp - min_val_tensor[:5]) / (max_val_tensor[:5] - min_val_tensor[:5])
    
    start = time.perf_counter_ns()
    res = solver.calc(inp)
    end = time.perf_counter_ns()
    
    elapsed_ns = end - start
    print(f"Elapsed time: {elapsed_ns} ns")
    print(f"Elapsed time: {elapsed_ns / 1e9:.6f} s")
    
    res[0] = res[0] * (max_val_tensor[5] - min_val_tensor[5]) + min_val_tensor[5]
    res[1] = res[1] * (max_val_tensor[6] - min_val_tensor[6]) + min_val_tensor[6]
    
    print(f"I want 3.815230 and -1.390400 and I get: {res[0]} and {res[1]}")
    '''

if __name__ == "__main__":
    main()
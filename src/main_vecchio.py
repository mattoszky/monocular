
from torch.utils.data import DataLoader, random_split, Subset
from utils.dataset import CustomDataset
import numpy as np
import torch
import torch.nn as nn
from solvers.solver import Solver
from solvers.solver_denorm import Solver_Denorm
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mn', type=str, required=True, help="Model name")
    args = parser.parse_args()
    
    #carico i dati
    file_path = "./../data/dataset/data20.dat"
    data = np.loadtxt(file_path)
    
    
    norm = True
    seed = 287
    torch.manual_seed(seed)
    batch_size = 32
    if norm:
        # Estrai input e ground truth
        inputs = data[:, :5]
        gt = data[:, 5:]
        
        total_size = len(inputs)
        train_size = int(0.7 * total_size)  # 70% per il training
        val_size = int(0.2 * total_size)    # 20% per la validazione

        # Genera indici casuali per suddividere i dati
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Dividi i dati in base agli indici
        train_inputs = inputs[train_indices]
        train_gt = gt[train_indices]
        val_inputs = inputs[val_indices]
        val_gt = gt[val_indices]
        test_inputs = inputs[test_indices]
        test_gt = gt[test_indices]

        # Calcola i minimi e massimi solo sul training set
        min_vals_input = train_inputs.min(axis=0)
        max_vals_input = train_inputs.max(axis=0)
        min_vals_gt = train_gt.min(axis=0)
        max_vals_gt = train_gt.max(axis=0)

        # Normalizza i set
        train_inputs = (train_inputs - min_vals_input) / (max_vals_input - min_vals_input)
        val_inputs = (val_inputs - min_vals_input) / (max_vals_input - min_vals_input)
        test_inputs = (test_inputs - min_vals_input) / (max_vals_input - min_vals_input)
        
        train_gt = (train_gt - min_vals_gt) / (max_vals_gt - min_vals_gt)
        val_gt = (val_gt - min_vals_gt) / (max_vals_gt - min_vals_gt)
        test_gt = (test_gt - min_vals_gt) / (max_vals_gt - min_vals_gt)
        
        # Converti min e max in torch.Tensors 
        max_val_tensor = torch.cat((torch.from_numpy(max_vals_input), torch.from_numpy(max_vals_gt)), dim=0)
        min_val_tensor = torch.cat((torch.from_numpy(min_vals_input), torch.from_numpy(min_vals_gt)), dim=0)

        train_dataset = CustomDataset(train_inputs, train_gt)
        val_dataset = CustomDataset(val_inputs, val_gt)
        test_dataset = CustomDataset(test_inputs, test_gt)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        solver = Solver(seed = seed,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    min_vals=min_val_tensor,
                    max_vals=max_val_tensor,
                    epochs=300,
                    lr=0.0001,
                    #train_criterion=nn.HuberLoss(delta=0.3), 
                    train_criterion=nn.MSELoss(),
                    #train_criterion=nn.SmoothL1Loss(),
                    test_criterion=nn.L1Loss(),
                    inc_val=30,
                    batch_size=batch_size,
                    checkpoint_path="../modelli",
                    model_name=args.mn, 
                    print_every=10,
                    th_x=10,
                    th_y=5
                )
        solver.load_model()
        solver.train()
        #solver.load_model()
        solver.test()
        
    else:     
        inputs = data[:, :5]
        gt = data[:, 5:]

        dataset = CustomDataset(inputs, gt)
        
        total_size = len(dataset)
        train_size = int(0.7 * total_size)  # 70% per il training
        val_size = int(0.2 * total_size)    # 20% per la validazione
        test_size = total_size - train_size - val_size  # 10% per il test

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        solver = Solver_Denorm(seed = seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=10,
                lr=0.0001, #0.0001 o 0.001
                criterion=nn.MSELoss(),
                inc_val=5,
                batch_size=batch_size,
                checkpoint_path="modelli",
                model_name=args.mn)
        #solver.load_model()
        #solver.train()
        #solver.load_model()
        #solver.test()
    
    #solver.save_model()

    """
    inp = torch.tensor([0.000000, 2500.000000, 839.000000, 117.000000, 167.000000], dtype=torch.float32)
    min_val_tensor = min_val_tensor.type(torch.float32)
    max_val_tensor = max_val_tensor.type(torch.float32)
    inp = (inp - min_val_tensor[:5]) / (max_val_tensor[:5] - min_val_tensor[:5])
    
    res = solver.calc(inp)
    
    print(f"min vals: {min_vals[5:]}")
    print(f"max vals: {max_vals[5:]}")
    print(f"res[0] norm: {res[0]}")
    print(f"res[1] norm: {res[1]}")
    
    res[0] = res[0] * (max_val_tensor[5] - min_val_tensor[5]) + min_val_tensor[5]
    res[1] = res[1] * (max_val_tensor[6] - min_val_tensor[6]) + min_val_tensor[6]
    
    
    #print(f"mi aspetto 3.815230 e -1.390400 e ottengo: {res[0]} e {res[1]}")
    
    
    """


if __name__ == "__main__":
    main()
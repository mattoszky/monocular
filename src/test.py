'''
seed = 100
lr = 0.001
loss = 'MSE'
delta = None
bs = 32

name = "256_" + str(seed) + "_" + loss + "_" + ((str(delta).replace('.', '_') + '_') if delta != None else "") + str(bs) + '_' + str(lr).replace('.', '_')

print(name)
'''

'''
import os
import torch
from nets.rete256 import Rete256
import numpy as np
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json

norm = True

th_x = 10
th_y = 5

device = "cuda"
epochs = 300
print_every = 10
inc_val = 30
test_criterion = nn.L1Loss()

model_name = "256_254_HL_0_3_32_0_00001"
train_criterion = nn.HuberLoss(delta=0.3)
#train_criterion = nn.HuberLoss(delta=0.8)
#train_criterion = nn.MSELoss()
#train_criterion = nn.SmoothL1Loss()
seed = 254
lr = 0.00001
stop_epoch = 300
batch_size = 32
torch.manual_seed(seed)
h, m, s = 2, 55, 4

file_path = "./../data/dataset/data.dat"
data = np.loadtxt(file_path)

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
max_vals = torch.cat((torch.from_numpy(max_vals_input), torch.from_numpy(max_vals_gt)), dim=0).to(device)
min_vals = torch.cat((torch.from_numpy(min_vals_input), torch.from_numpy(min_vals_gt)), dim=0).to(device)

train_dataset = CustomDataset(train_inputs, train_gt)
val_dataset = CustomDataset(val_inputs, val_gt)
test_dataset = CustomDataset(test_inputs, test_gt)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


net = Rete256()
checkpoint_path = "../modelli"


# function to load the model
check_path = os.path.join(checkpoint_path, model_name)
net.load_state_dict(torch.load(check_path))
print("Model loaded!")

net.to(device)

writer_info = SummaryWriter('./../runs/' + model_name + '/info')

test_loss_norm = 0.0
test_loss_denorm = 0.0
n_cones = 0
n_big_cones = 0
test_loss_norm_big_cones = 0
test_loss_norm_cones = 0
test_loss_denorm_big_cones = 0
test_loss_denorm_cones = 0
test_loss_denorm_x = 0
test_loss_denorm_y = 0
test_loss_denorm_d = 0
test_loss_denorm_x_cones = 0
test_loss_denorm_y_cones = 0
test_loss_denorm_d_cones = 0
test_loss_denorm_x_big_cones = 0
test_loss_denorm_y_big_cones = 0
test_loss_denorm_d_big_cones = 0
n_cones_near = 0
test_loss_denorm_d_cones_near = 0
net.eval()
with torch.no_grad():
    for batch_inputs, batch_gt in test_loader:
        batch_inputs, batch_gt = batch_inputs.to(device), batch_gt.to(device)
        outputs = net(batch_inputs)
        loss_norm = test_criterion(outputs, batch_gt)
        outputs = outputs * (max_vals[5:] - min_vals[5:]) + min_vals[5:]
        batch_gt = batch_gt * (max_vals[5:] - min_vals[5:]) + min_vals[5:]
        loss_denorm = test_criterion(outputs, batch_gt)
        test_loss_norm += loss_norm.item()
        test_loss_denorm += loss_denorm.item()
        x_out, y_out = outputs[:,0], outputs[:,1]
        x_gt, y_gt = batch_gt[:,0], batch_gt[:,1]
        d_out = (x_out**2 + y_out**2)**(1/2)
        d_gt = (x_gt**2 + y_gt**2)**(1/2)
        loss_x = test_criterion(x_out, x_gt)
        loss_y = test_criterion(y_out, y_gt)
        loss_d = test_criterion(d_out, d_gt)
        test_loss_denorm_x += loss_x.item()
        test_loss_denorm_y += loss_y.item()
        test_loss_denorm_d += loss_d.item()
        for single_input in batch_inputs:
            if (single_input[0] == 0):
                n_cones += 1
                test_loss_norm_cones += loss_norm.item()
                test_loss_denorm_cones += loss_denorm.item()
                test_loss_denorm_x_cones += loss_x.item()
                test_loss_denorm_y_cones += loss_y.item()
                test_loss_denorm_d_cones += loss_d.item()
            else:
                n_big_cones += 1
                test_loss_norm_big_cones += loss_norm.item()
                test_loss_denorm_big_cones += loss_denorm.item()
                test_loss_denorm_x_big_cones += loss_x.item()
                test_loss_denorm_y_big_cones += loss_y.item()
                test_loss_denorm_d_big_cones += loss_d.item()
        for single_gt in batch_gt:
            if single_gt[0] <= th_x and (single_gt[1] <= th_y/2 and single_gt[1] >= -th_y/2):
                n_cones_near += 1
                test_loss_denorm_d_cones_near += loss_d.item()
                

print(f"Test Loss norm: {test_loss_norm/len(test_loader):.10f}, Test Loss denorm: {test_loss_denorm/len(test_loader):.10f}")
print(f"Test loss x: {test_loss_denorm_x/len(test_loader):.10f}, Test loss y: {test_loss_denorm_y/len(test_loader):.10f}")
print(f"Test loss distance: {test_loss_denorm_d/len(test_loader):.10f}")

print(f"Test loss x cones: {test_loss_denorm_x_cones/n_cones:.10f}, Test loss y cones: {test_loss_denorm_y_cones/n_cones:.10f}")
print(f"Test loss distance cones: {test_loss_denorm_d_cones/n_cones:.10f}")

print(f"Test loss x big cones: {test_loss_denorm_x_big_cones/n_big_cones:.10f}, Test loss y big cones: {test_loss_denorm_y_big_cones/n_big_cones:.10f}")
print(f"Test loss distance big cones: {test_loss_denorm_d_big_cones/n_big_cones:.10f}")

print(f"Test loss distance near cones: {test_loss_denorm_d_cones_near/n_cones_near:.10f}")

net.train()
'''
import os
import torch
import numpy as np
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
writer_info = SummaryWriter('./../runs/' + "512_199_MSE_32_0_0001" + '/info')

markdown_table = f"""
| Metric  | Value |
|-------|----|
| model name |  {"512_199_MSE_32_0_0001"} |
| # of neurons in first layer   |  512 |
| batch size |  32 |
| seed |  199 |
| lr |  0.0001 |
| train criterion |  MSELoss |
| delta |  {"None"} |
| test criterion |  L1Loss |
| epochs |  300 |
| stop epoch |  245 |
| # of not inc |  30 |
| print every |  10 |
| training time |  "2 hours, 59 minutes e 40 seconds" |
| test loss denorm |  0.3349150859154036 |
| test loss denorm small cones |  0.33496200191284775 |
| # of small cones |  66939 |
| test loss denorm big cones|  0.33323848783353555 |
| # of big cones |  1818 |
| loss denorm x |  0.4758887684263858 |
| loss denorm x small cones |  0.4759808099227646 |
| loss denorm x big cones |  0.4726278956381216 |
| loss denorm y |  0.1939414034044215 |
| loss denorm y small cones |  0.19394319390291323 |
| loss denorm y big cones |  0.1938490800289488 |
| loss denorm d |  0.5221609153089412 |
| loss denorm d small cones |  0.52226090338015 |
| loss denorm d big cones |  0.518607454723318 |
| loss denorm d near cones (10, +-2.5) |  0.5105377173626491 |
| # of near cones |  24138 |
"""


# Scrittura della tabella nel SummaryWriter
writer_info.add_text("Results", markdown_table, 0)
#writer_info.add_text("info", json.dumps(data_dict, indent=4), 1)  
writer_info.flush()
writer_info.close()

'''

import time

def get_time(elapsed):
    """
    Returns h, m, s from a given elapsed time
    """
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return h, m, s

start = time.time()
for i in range(100000000):
    None
stop = time.time()

elapsed = None

h, m, s = get_time(elapsed) if elapsed else (0, 0, 0)

print(h)
print(m)
print(s)
'''
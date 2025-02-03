import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import time

import PlotRes
from Rete import Rete

def get_time(elapsed):
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return h, m, s

def print_time(elapsed, stop_epoch):
    hours, minutes, seconds = get_time(elapsed)
    print(f"stop_epoch: {stop_epoch}")
    print(f"Il training è durato {hours} ore, {minutes} minuti e {seconds} secondi")
    
def save_info_model(elapsed, checkpoint_path, model_name, batch_size, seed, lr, epochs, stop_epoch, inc_val, test_loss_norm, test_loss_denorm, len_test, n_neurons, n_cones, n_big_cones, test_loss_denorm_cones, test_loss_denorm_big_cones, test_loss_norm_cones, test_loss_norm_big_cones):
    h, m, s = get_time(elapsed)
    with open(checkpoint_path + "/res.txt", "a") as f:
        f.write(f"{model_name}\n{{\n" +
                f"\tnumero di neuroni nel primo strato: {n_neurons},\n" +
                f"\tbatch size: {batch_size},\n" +
                f"\tseed: {seed},\n" +
                f"\tlr: {lr},\n" +
                f"\tepoche: {epochs},\n" +
                f"\tepocha di terminazione: {stop_epoch},\n" +
                f"\tnumero di incrementi max nel validation set: {inc_val},\n" +
                f"\ttest loss norm: {test_loss_norm / len_test:.10f},\n" + 
                f"\ttest loss denorm: {test_loss_denorm / len_test:.10f},\n" + 
                f"\ttrain time: {h} ore, {m} minuti e {s} secondi\n\n" + 
                f"\tnumero coni 0: {n_cones},\n" +
                f"\tnumero coni 1: {n_big_cones},\n" +
                f"\tloss norm coni 0: {test_loss_norm_cones/n_cones:.10f},\n" +
                f"\tloss norm coni 1: {test_loss_norm_big_cones/n_big_cones:.10f},\n" +
                f"\tloss denorm coni 0: {test_loss_denorm_cones/n_cones:.10f},\n" +
                f"\tloss denorm coni 1: {test_loss_denorm_big_cones/n_big_cones:.10f}\n}}\n\n")
    

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, seed, train_loader, val_loader, test_loader, epochs, lr, criterion, inc_val, batch_size, checkpoint_path, model_name, min_vals=None, max_vals=None):
        self.seed = seed #da togliere poi nella versione finale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Rete().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.min_vals = min_vals.to(self.device)
        self.max_vals = max_vals.to(self.device)
        self.inc_val = inc_val
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name

    def save_model(self):
        # if you want to save the model
        check_path = os.path.join(self.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        check_path = os.path.join(self.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print("Model loaded!")
    
    def train(self):
        self.net.train()
        y_train_norm = np.empty(self.epochs)
        y_val_norm = np.empty(self.epochs)
        y_train_denorm = np.empty(self.epochs)
        y_val_denorm = np.empty(self.epochs)
        n_plus = 0
        stop_epoch = 0
        start = time.time()
        for epoch in range(self.epochs):
            stop_epoch = epoch + 1
            running_loss_norm = 0.0
            running_loss_denorm = 0.0
            for batch_inputs, batch_gt in self.train_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(batch_inputs)
                loss_norm = self.criterion(outputs, batch_gt)
                loss_norm.backward()
                self.optimizer.step()
                
                outputs = outputs * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                batch_gt = batch_gt * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                loss_denorm = self.criterion(outputs, batch_gt)
                running_loss_norm += loss_norm.item()
                running_loss_denorm += loss_denorm.item()
                
            val_loss_norm = 0.0
            val_loss_denorm = 0.0
            with torch.no_grad():
                for batch_inputs, batch_gt in self.val_loader:
                    batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                    outputs = self.net(batch_inputs)
                    loss_norm = self.criterion(outputs, batch_gt)
                    outputs = outputs * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                    batch_gt = batch_gt * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                    loss_denorm = self.criterion(outputs, batch_gt)
                    val_loss_norm += loss_norm.item()
                    val_loss_denorm += loss_denorm.item()

            y_val_norm[epoch] = val_loss_norm/len(self.val_loader)
            y_train_norm[epoch] = running_loss_norm/len(self.train_loader)
            y_val_denorm[epoch] = val_loss_denorm/len(self.val_loader)
            y_train_denorm[epoch] = running_loss_denorm/len(self.train_loader)
            if ((epoch+1) % 10 == 0):
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss norm: {y_train_norm[epoch]:.10f}, Train Loss denorm: {y_train_denorm[epoch]:.10f}, Val Loss norm: {y_val_norm[epoch]:.10f}, Val Loss denorm: {y_val_denorm[epoch]:.10f}")
            if (epoch > 0):
                if (y_val_norm[epoch] > y_val_norm[epoch-1]):
                    n_plus = n_plus + 1
                else:
                    n_plus = 0
            if (n_plus >= self.inc_val):
                print(f"Last Epoch {epoch+1}/{self.epochs}, Train Loss norm: {y_train_norm[epoch]:.10f}, Train Loss denorm: {y_train_denorm[epoch]:.10f}, Val Loss norm: {y_val_norm[epoch]:.10f}, Val Loss denorm: {y_val_denorm[epoch]:.10f}")
                break
        end = time.time()
        self.elapsed = end-start
        self.stop_epoch = stop_epoch
        print_time(self.elapsed, stop_epoch)
        PlotRes.plot_results(y_train_norm, y_val_norm, y_train_denorm, y_val_denorm, stop_epoch, "grafici/training/")
        if (n_plus >= 2):  
            print('Finished Training because of Validation') 
        else:
            print('Finished Training')
            
    def train_denorm(self):
        self.net.train()
        y_train_denorm = np.empty(self.epochs)
        y_val_denorm = np.empty(self.epochs)
        n_plus = 0
        stop_epoch = 0
        start = time.time()
        for epoch in range(self.epochs):
            stop_epoch = epoch + 1
            running_loss_denorm = 0.0
            for batch_inputs, batch_gt in self.train_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(batch_inputs)
                loss_denorm = self.criterion(outputs, batch_gt)
                loss_denorm.backward()
                self.optimizer.step()
                running_loss_denorm += loss_denorm.item()
                
            val_loss_denorm = 0.0
            with torch.no_grad():
                for batch_inputs, batch_gt in self.val_loader:
                    batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                    outputs = self.net(batch_inputs)
                    loss_denorm = self.criterion(outputs, batch_gt)
                    val_loss_denorm += loss_denorm.item()

            y_val_denorm[epoch] = val_loss_denorm/len(self.val_loader)
            y_train_denorm[epoch] = running_loss_denorm/len(self.train_loader)
            if ((epoch+1) % 10 == 0):
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss denorm: {y_train_denorm[epoch]:.10f}, Val Loss denorm: {y_val_denorm[epoch]:.10f}")
            if (epoch > 0):
                if (y_val_denorm[epoch] > y_val_denorm[epoch-1]):
                    n_plus = n_plus + 1
                else:
                    n_plus = 0
            if (n_plus >= self.inc_val):
                print(f"Last Epoch {epoch+1}/{self.epochs}, Train Loss denorm: {y_train_denorm[epoch]:.10f}, Val Loss denorm: {y_val_denorm[epoch]:.10f}")
                break
        end = time.time()
        self.elapsed = end-start
        self.stop_epoch = stop_epoch
        print_time(self.elapsed, stop_epoch)
        PlotRes.plot_results([], [], y_train_denorm, y_val_denorm, stop_epoch, "grafici/training/", False)
        if (n_plus >= 2):  
            print('Finished Training because of Validation') 
        else:
            print('Finished Training')
    
    def test(self):
        test_loss_norm = 0.0
        test_loss_denorm = 0.0
        n_cones = 0
        n_big_cones = 0
        test_loss_norm_big_cones = 0
        test_loss_norm_cones = 0
        test_loss_denorm_big_cones = 0
        test_loss_denorm_cones = 0
        self.net.eval()
        with torch.no_grad():
            for batch_inputs, batch_gt in self.test_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss_norm = self.criterion(outputs, batch_gt)
                outputs = outputs * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                batch_gt = batch_gt * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                """
                if (test_loss_denorm == 0):
                    print("Ho predetto:")
                    print(outputs)
                    print("Mi aspettavo:")
                    print(batch_gt)
                """
                loss_denorm = self.criterion(outputs, batch_gt)
                test_loss_norm += loss_norm.item()
                test_loss_denorm += loss_denorm.item()
                for single_input in batch_inputs:
                    if (single_input[0] == 0):
                        n_cones += 1
                        test_loss_norm_cones += loss_norm.item()
                        test_loss_denorm_cones += loss_denorm.item()
                    else:
                        n_big_cones += 1
                        test_loss_norm_big_cones += loss_norm.item()
                        test_loss_denorm_big_cones += loss_denorm.item()

        
        print(f"Test Loss norm: {test_loss_norm/len(self.test_loader):.10f}, Test Loss denorm: {test_loss_denorm/len(self.test_loader):.10f}")
        save_info_model(self.elapsed, self.checkpoint_path, self.model_name, self.batch_size, self.seed, self.lr, self.epochs, self.stop_epoch, self.inc_val, test_loss_norm, test_loss_denorm, len(self.test_loader), self.net.layer1[0].out_features, n_cones, n_big_cones, test_loss_denorm_cones, test_loss_denorm_big_cones, test_loss_norm_cones, test_loss_norm_big_cones)
        self.net.train()
        
    def test_denorm(self):
        test_loss_denorm = 0.0
        n_cones = 0
        n_big_cones = 0
        test_loss_denorm_big_cones = 0
        test_loss_denorm_cones = 0
        self.net.eval()
        with torch.no_grad():
            for batch_inputs, batch_gt in self.test_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss_denorm = self.criterion(outputs, batch_gt)
                test_loss_denorm += loss_denorm.item()
                for single_input in batch_inputs:
                    if (single_input[0] == 0):
                        n_cones += 1
                        test_loss_denorm_cones += loss_denorm.item()
                    else:
                        n_big_cones += 1
                        test_loss_denorm_big_cones += loss_denorm.item()

        
        print(f"Test Loss denorm: {test_loss_denorm/len(self.test_loader):.10f}")
        save_info_model(self.elapsed, self.checkpoint_path, self.model_name, self.batch_size, self.seed, self.lr, self.epochs, self.stop_epoch, self.inc_val, 1, test_loss_denorm, len(self.test_loader), self.net.layer1[0].out_features, n_cones, n_big_cones, test_loss_denorm_cones, test_loss_denorm_big_cones, 1, 1)
        self.net.train()
    
    def calc(self, inputs):
        inputs = inputs.to(self.device)
        self.net.eval()
        out = self.net(inputs)
        self.net.train()
        out = out.to("cpu")
        return out
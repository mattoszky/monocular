import torch
import torch.optim as optim
import os
import numpy as np
import time
import json

from torch.utils.tensorboard import SummaryWriter

from nets.rete256 import Rete256

def get_time(elapsed):
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return h, m, s

def print_time(elapsed, stop_epoch):
    hours, minutes, seconds = get_time(elapsed)
    print(f"stop_epoch: {stop_epoch}")
    print(f"Il training è durato {hours} ore, {minutes} minuti e {seconds} secondi")    

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, seed, train_loader, val_loader, test_loader, epochs, lr, train_criterion, test_criterion, inc_val, batch_size, checkpoint_path, model_name, min_vals, max_vals, print_every, th_x=5, th_y=5):
        self.seed = seed #da togliere poi nella versione finale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Rete256().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.train_criterion = train_criterion
        self.test_criterion = test_criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.min_vals = min_vals.to(self.device)
        self.max_vals = max_vals.to(self.device)
        self.inc_val = inc_val
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.print_every = print_every
        self.th_x = th_x
        self.th_y = th_y
        self.writer_train = SummaryWriter('./../runs/' + self.model_name + '/train')
        self.writer_val = SummaryWriter('./../runs/' + self.model_name + '/val')
        self.writer_info = SummaryWriter('./../runs/' + self.model_name + '/info')

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
                loss_norm = self.train_criterion(outputs, batch_gt)
                loss_norm.backward()
                self.optimizer.step()
                running_loss_norm += loss_norm.item()
                # mi salvo la loss denormalizzata ogni 10 epoche per avere un riscontro
                if ((epoch+1) % 10 == 0):
                    outputs = outputs * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                    batch_gt = batch_gt * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                    loss_denorm = self.train_criterion(outputs, batch_gt)
                    running_loss_denorm += loss_denorm.item()
            self.writer_train.add_scalar('loss', running_loss_norm/len(self.train_loader), epoch+1)
            val_loss_norm, val_loss_denorm = self.validation(epoch)
            if ((epoch+1) % self.print_every == 0):
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss norm: {running_loss_norm/len(self.train_loader):.10f}, Train Loss denorm: {running_loss_denorm/len(self.train_loader):.10f}, Val Loss norm: {val_loss_norm:.10f}, Val Loss denorm: {val_loss_denorm:.10f}")
            if epoch == 0:
                self.best_val_loss = val_loss_norm
                n_plus = 0
                self.save_model()
            else:
                if (val_loss_norm > self.best_val_loss):
                    n_plus = n_plus + 1
                else:
                    n_plus = 0
                    self.best_val_loss = val_loss_norm
                    self.save_model()
            if (n_plus >= self.inc_val):
                print(f"Last Epoch {epoch+1}/{self.epochs}, Train Loss norm: {running_loss_norm/len(self.train_loader):.10f}, Train Loss denorm: {running_loss_denorm/len(self.train_loader):.10f}, Val Loss norm: {val_loss_norm:.10f}, Val Loss denorm: {val_loss_denorm:.10f}")
                break
            self.writer_train.flush()
            self.writer_val.flush()
        end = time.time()
        self.elapsed = end-start
        self.stop_epoch = stop_epoch
        print_time(self.elapsed, stop_epoch)
        if (n_plus >= 2):  
            print('Finished Training because of Validation') 
        else:
            print('Finished Training')
        self.writer_train.close()
        self.writer_val.close()
        
    def validation(self, epoch):
        self.net.eval()
        val_loss_norm = 0.0
        val_loss_denorm = 0.0
        with torch.no_grad():
            for batch_inputs, batch_gt in self.val_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss_norm = self.train_criterion(outputs, batch_gt)
                val_loss_norm += loss_norm.item()
                if ((epoch+1) % 10 == 0):
                    outputs = outputs * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                    batch_gt = batch_gt * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                    loss_denorm = self.train_criterion(outputs, batch_gt)
                    val_loss_denorm += loss_denorm.item()
        self.writer_val.add_scalar('loss', val_loss_norm/len(self.val_loader), epoch+1)
        self.net.train()
        return val_loss_norm/len(self.val_loader), val_loss_denorm/len(self.val_loader)
    
    def test(self):
        test_loss_norm = 0.0
        test_loss_denorm = 0.0
        self.n_cones = 0
        self.n_big_cones = 0
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
        self.net.eval()
        with torch.no_grad():
            for batch_inputs, batch_gt in self.test_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss_norm = self.test_criterion(outputs, batch_gt)
                outputs = outputs * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                batch_gt = batch_gt * (self.max_vals[5:] - self.min_vals[5:]) + self.min_vals[5:]
                loss_denorm = self.test_criterion(outputs, batch_gt)
                test_loss_norm += loss_norm.item()
                test_loss_denorm += loss_denorm.item()
                x_out, y_out = outputs[:,0], outputs[:,1]
                x_gt, y_gt = batch_gt[:,0], batch_gt[:,1]
                d_out = (x_out**2 + y_out**2)**(1/2)
                d_gt = (x_gt**2 + y_gt**2)**(1/2)
                loss_x = self.test_criterion(x_out, x_gt)
                loss_y = self.test_criterion(y_out, y_gt)
                loss_d = self.test_criterion(d_out, d_gt)
                test_loss_denorm_x += loss_x.item()
                test_loss_denorm_y += loss_y.item()
                test_loss_denorm_d += loss_d.item()
                for single_input in batch_inputs:
                    if (single_input[0] == 0):
                        self.n_cones += 1
                        test_loss_norm_cones += loss_norm.item()
                        test_loss_denorm_cones += loss_denorm.item()
                        test_loss_denorm_x_cones += loss_x.item()
                        test_loss_denorm_y_cones += loss_y.item()
                        test_loss_denorm_d_cones += loss_d.item()
                    else:
                        self.n_big_cones += 1
                        test_loss_norm_big_cones += loss_norm.item()
                        test_loss_denorm_big_cones += loss_denorm.item()
                        test_loss_denorm_x_big_cones += loss_x.item()
                        test_loss_denorm_y_big_cones += loss_y.item()
                        test_loss_denorm_d_big_cones += loss_d.item()
                for single_gt in batch_gt:
                    if single_gt[0] <= self.th_x and (single_gt[1] <= self.th_y and single_gt[1] >= -self.th_y):
                        n_cones_near += 1
                        test_loss_denorm_d_cones_near += loss_d.item()
                        
        
        print(f"Test Loss norm: {test_loss_norm/len(self.test_loader):.10f}, Test Loss denorm: {test_loss_denorm/len(self.test_loader):.10f}")
        print(f"Test loss x: {test_loss_denorm_x/len(self.test_loader):.10f}, Test loss y: {test_loss_denorm_y/len(self.test_loader):.10f}")
        print(f"Test loss distance: {test_loss_denorm_d/len(self.test_loader):.10f}")
        
        print(f"Test loss x cones: {test_loss_denorm_x_cones/self.n_cones:.10f}, Test loss y cones: {test_loss_denorm_y_cones/self.n_cones:.10f}")
        print(f"Test loss distance cones: {test_loss_denorm_d_cones/self.n_cones:.10f}")
        
        print(f"Test loss x big cones: {test_loss_denorm_x_big_cones/self.n_big_cones:.10f}, Test loss y big cones: {test_loss_denorm_y_big_cones/self.n_big_cones:.10f}")
        print(f"Test loss distance big cones: {test_loss_denorm_d_big_cones/self.n_big_cones:.10f}")
        
        print(f"Test loss distance near cones: {test_loss_denorm_d_cones_near/n_cones_near:.10f}")
        
        self.save_info_model(test_loss_norm, 
                        test_loss_denorm, 
                        test_loss_denorm_cones, 
                        test_loss_denorm_big_cones, 
                        test_loss_norm_cones, 
                        test_loss_norm_big_cones,
                        
                        test_loss_denorm_x/len(self.test_loader),
                        test_loss_denorm_y/len(self.test_loader),
                        test_loss_denorm_d/len(self.test_loader),
                        test_loss_denorm_x_cones/self.n_cones,
                        test_loss_denorm_y_cones/self.n_cones,
                        test_loss_denorm_d_cones/self.n_cones,
                        test_loss_denorm_x_big_cones/self.n_big_cones,
                        test_loss_denorm_y_big_cones/self.n_big_cones,
                        test_loss_denorm_d_big_cones/self.n_big_cones,
                        test_loss_denorm_d_cones_near/n_cones_near
                        )
        
        self.net.train()
    
    def calc(self, inputs):
        inputs = inputs.to(self.device)
        self.net.eval()
        out = self.net(inputs)
        self.net.train()
        out = out.to("cpu")
        return out
    
    def save_info_model(self, test_loss_norm, 
                        test_loss_denorm, 
                        test_loss_denorm_cones, 
                        test_loss_denorm_big_cones, 
                        test_loss_norm_cones, 
                        test_loss_norm_big_cones,
                        
                        test_loss_denorm_x,
                        test_loss_denorm_y,
                        test_loss_denorm_d,
                        test_loss_denorm_x_cones,
                        test_loss_denorm_y_cones,
                        test_loss_denorm_d_cones,
                        test_loss_denorm_x_big_cones,
                        test_loss_denorm_y_big_cones,
                        test_loss_denorm_d_big_cones,
                        test_loss_denorm_d_cones_near
                        ):
        h, m, s = get_time(self.elapsed)
        data_dict = {
            "model name": self.model_name,
            "numero di neuroni nel primo strato": self.net.layer1[0].out_features,
            "batch size": self.batch_size, 
            "seed": self.seed, 
            "lr": self.lr,
            "criterion train": self.train_criterion.__class__.__name__,
            "delta": self.train_criterion.delta if self.train_criterion.__class__.__name__ == "HuberLoss" else "None",
            "criterion test": self.test_criterion.__class__.__name__,
            "epoche": self.epochs, 
            "epocha di terminazione": self.stop_epoch, 
            "numero di non miglioramenti max nel validation set": self.inc_val,
            "stampa info ogni": self.print_every,
            "test loss norm": (test_loss_norm / len(self.test_loader)), 
            "test loss denorm": (test_loss_denorm / len(self.test_loader)),
            "tempo di addestramento": f"{h} ore, {m} minuti e {s} secondi",
            "numero coni 0": self.n_cones,
            "numero coni 1": self.n_big_cones,
            "loss norm coni 0": (test_loss_norm_cones/self.n_cones),
            "loss norm coni 1": (test_loss_norm_big_cones/self.n_big_cones),
            "loss denorm coni 0": (test_loss_denorm_cones/self.n_cones),
            "loss denorm coni 1": (test_loss_denorm_big_cones/self.n_big_cones),
            
            "loss denorm x" : test_loss_denorm_x,
            "loss denorm y" : test_loss_denorm_y,
            "loss denorm d" : test_loss_denorm_d,
            "loss denorm x coni 0" : test_loss_denorm_x_cones,
            "loss denorm y coni 0" : test_loss_denorm_y_cones,
            "loss denorm d coni 0" : test_loss_denorm_d_cones,
            "loss denorm x coni 1" : test_loss_denorm_x_big_cones,
            "loss denorm y coni 1" : test_loss_denorm_y_big_cones,
            "loss denorm d coni 1" : test_loss_denorm_d_big_cones,
            f"loss denorm d coni vicini ({self.th_x}, +-{self.th_y})" : test_loss_denorm_d_cones_near,
            }
        self.writer_info.add_text("info", json.dumps(data_dict, indent=4), 1)  
        self.writer_info.flush()
        self.writer_info.close()
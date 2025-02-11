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

class Solver_Denorm(object):
    """Solver for training and testing."""

    def __init__(self, seed, train_loader, val_loader, test_loader, epochs, lr, criterion, inc_val, batch_size, checkpoint_path, model_name):
        self.seed = seed #da togliere poi nella versione finale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Rete256().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.inc_val = inc_val
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.writer_train = SummaryWriter('./runs/' + self.model_name + '/train')
        self.writer_val = SummaryWriter('./runs/' + self.model_name + '/val')
        self.writer_info = SummaryWriter('./runs/' + self.model_name + '/info')

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
                
    def save_info_model(self, test_loss, test_loss_cones, test_loss_big_cones):
        h, m, s = get_time(self.elapsed)
        data_dict = {
            "model name": self.model_name,
            "numero di neuroni nel primo strato": self.net.layer1[0].out_features,
            "batch size": self.batch_size, 
            "seed": self.seed, 
            "lr": self.lr,
            "epoche": self.epochs, 
            "epocha di terminazione": self.stop_epoch, 
            "numero di incrementi max nel validation set": self.inc_val,
            "test loss": (test_loss / len(self.test_loader)), 
            "tempo di addestramento": f"{h} ore, {m} minuti e {s} secondi",
            "numero coni 0": self.n_cones,
            "numero coni 1": self.n_big_cones,
            "loss coni 0": (test_loss_cones/self.n_cones),
            "loss coni 1": (test_loss_big_cones/self.n_big_cones),
            }
        self.writer_info.add_text("info", json.dumps(data_dict, indent=4), 1)  
        self.writer_info.flush()
        self.writer_info.close()
    
    def calc(self, inputs):
        inputs = inputs.to(self.device)
        self.net.eval()
        out = self.net(inputs)
        self.net.train()
        out = out.to("cpu")
        return out
        
    def train(self):
        self.net.train()
        n_plus = 0
        stop_epoch = 0
        start = time.time()
        for epoch in range(self.epochs):
            stop_epoch = epoch + 1
            running_loss = 0.0
            for batch_inputs, batch_gt in self.train_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(batch_inputs)
                loss = self.criterion(outputs, batch_gt)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.writer_train.add_scalar('loss', running_loss/len(self.train_loader), epoch+1)
            val_loss = self.validation(epoch)
            running_loss = running_loss/len(self.train_loader)
            if ((epoch+1) % 10 == 0):
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {running_loss:.10f}, Val Loss: {val_loss:.10f}")
            if (epoch > 0):
                if (val_loss/len(self.val_loader) > old_val_loss):
                    n_plus = n_plus + 1
                else:
                    n_plus = 0
            if (n_plus >= self.inc_val):
                print(f"Last Epoch {epoch+1}/{self.epochs}, Train Loss : {running_loss:.10f}, Val Loss : {val_loss:.10f}")
                break
            self.writer_train.flush()
            self.writer_val.flush()
            old_val_loss = val_loss
        end = time.time()
        self.elapsed = end-start
        self.stop_epoch = stop_epoch
        print_time(self.elapsed, stop_epoch)
        if (n_plus >= 2):  
            print('Finished Training because of Validation') 
        else:
            print('Finished Training')
    
    def validation(self, epoch):
        self.net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_gt in self.val_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss = self.criterion(outputs, batch_gt)
                val_loss += loss.item()
        self.writer_val.add_scalar('loss', val_loss/len(self.val_loader), epoch+1)
        self.net.train()
        return val_loss/len(self.val_loader)
    
    def test(self):
        test_loss = 0.0
        self.n_cones = 0
        self.n_big_cones = 0
        test_loss_big_cones = 0
        test_loss_cones = 0
        self.net.eval()
        with torch.no_grad():
            for batch_inputs, batch_gt in self.test_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss = self.criterion(outputs, batch_gt)
                test_loss += loss.item()
                for single_input in batch_inputs:
                    if (single_input[0] == 0):
                        self.n_cones += 1
                        test_loss_cones += loss.item()
                    else:
                        self.n_big_cones += 1
                        test_loss_big_cones += loss.item()

        print(f"Test Loss: {test_loss/len(self.test_loader):.10f}")
        self.save_info_model(test_loss, 
                        test_loss_cones, 
                        test_loss_big_cones
                        )
        self.net.train()
    
    
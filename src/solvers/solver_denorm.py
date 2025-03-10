import torch
import torch.optim as optim
import os
import time

from torch.utils.tensorboard import SummaryWriter

from nets.net256 import Net256
from nets.net512 import Net512

def get_time(elapsed):
    """
    Returns h, m, s from a given elapsed time
    """
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return h, m, s

def print_time(elapsed, stop_epoch):
    hours, minutes, seconds = get_time(elapsed)
    print(f"stop_epoch: {stop_epoch}")
    print(f"Training time: {hours} hours, {minutes} minutes and {seconds} seconds")  
    
class Solver_Denorm(object):
    """
    Solver for training and testing.
    """

    def __init__(self, seed, train_loader, val_loader, test_loader, epochs, lr, train_criterion, test_criterion, inc_val, batch_size, checkpoint_path, model_name, print_every, th_x, th_y):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net256().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.train_criterion = train_criterion
        self.test_criterion = test_criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.inc_val = inc_val
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name + "_DENORM"
        self.print_every = print_every
        self.th_x = th_x
        self.th_y = th_y
        self.writer_train = SummaryWriter('./../runs/' + self.model_name + '/train')
        self.writer_val = SummaryWriter('./../runs/' + self.model_name + '/val')
        self.writer_info = SummaryWriter('./../runs/' + self.model_name + '/info')

    def save_model(self):
        """
        Saves the model with the model name in the provided path
        """
        
        check_path = os.path.join(self.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        """
        Gets the model with the model name in the provided path
        """
        
        check_path = os.path.join(self.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print("Model loaded!")
        
    def train(self):
        """
        Train
        """
        
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
                loss = self.train_criterion(outputs, batch_gt)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.writer_train.add_scalar('loss', running_loss/len(self.train_loader), epoch+1)
            val_loss = self.validation(epoch)
            running_loss = running_loss/len(self.train_loader)
            if ((epoch+1) % self.print_every == 0):
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {running_loss:.10f}, Val Loss: {val_loss:.10f}")
            if epoch == 0:
                self.best_val_loss = val_loss
                n_plus = 0
                self.save_model()
            else:
                if (val_loss > self.best_val_loss):
                    n_plus = n_plus + 1
                else:
                    n_plus = 0
                    self.best_val_loss = val_loss
                    self.save_model()
            if (n_plus >= self.inc_val):
                print(f"Last Epoch {epoch+1}/{self.epochs}, Train Loss : {running_loss:.10f}, Val Loss : {val_loss:.10f}")
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
        """
        Validation
        """
        
        self.net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_gt in self.val_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss = self.train_criterion(outputs, batch_gt)
                val_loss += loss.item()
        self.writer_val.add_scalar('loss', val_loss/len(self.val_loader), epoch+1)
        self.net.train()
        return val_loss/len(self.val_loader)
    
    def test(self):
        """
        Testing
        """
        self.load_model()
        test_loss = 0.0
        self.n_cones = 0.0
        self.n_big_cones = 0.0
        self.n_cones_near = 0.0
        test_loss_big_cones = 0.0
        test_loss_cones = 0.0
        test_loss_x = 0.0
        test_loss_y = 0.0
        test_loss_d = 0.0
        test_loss_x_cones = 0.0
        test_loss_y_cones = 0.0
        test_loss_d_cones = 0.0
        test_loss_x_big_cones = 0.0
        test_loss_y_big_cones = 0.0
        test_loss_d_big_cones = 0.0
        test_loss_d_cones_near = 0.0
        self.net.eval()
        with torch.no_grad():
            for batch_inputs, batch_gt in self.test_loader:
                batch_inputs, batch_gt = batch_inputs.to(self.device), batch_gt.to(self.device)
                outputs = self.net(batch_inputs)
                loss = self.test_criterion(outputs, batch_gt)
                test_loss += loss.item()
                x_out, y_out = outputs[:,0], outputs[:,1]
                x_gt, y_gt = batch_gt[:,0], batch_gt[:,1]
                d_out = (x_out**2 + y_out**2)**(1/2)
                d_gt = (x_gt**2 + y_gt**2)**(1/2)
                loss_x = self.test_criterion(x_out, x_gt)
                loss_y = self.test_criterion(y_out, y_gt)
                loss_d = self.test_criterion(d_out, d_gt)
                test_loss_x += loss_x.item()
                test_loss_y += loss_y.item()
                test_loss_d += loss_d.item()
                for single_input in batch_inputs:
                    if (single_input[0] == 0):
                        self.n_cones += 1
                        test_loss_cones += loss.item()
                        test_loss_x_cones += loss_x.item()
                        test_loss_y_cones += loss_y.item()
                        test_loss_d_cones += loss_d.item()
                    else:
                        self.n_big_cones += 1
                        test_loss_big_cones += loss.item()
                        test_loss_x_big_cones += loss_x.item()
                        test_loss_y_big_cones += loss_y.item()
                        test_loss_d_big_cones += loss_d.item()
                for single_gt in batch_gt:
                    if single_gt[0] <= self.th_x and (single_gt[1] <= self.th_y/2 and single_gt[1] >= -self.th_y/2):
                        self.n_cones_near += 1
                        test_loss_d_cones_near += loss_d.item()

        self.save_info_model(
                        test_loss/len(self.test_loader), 
                        test_loss_cones/self.n_cones, 
                        test_loss_big_cones/self.n_big_cones, 
                        test_loss_x/len(self.test_loader),
                        test_loss_y/len(self.test_loader),
                        test_loss_d/len(self.test_loader),
                        test_loss_x_cones/self.n_cones,
                        test_loss_y_cones/self.n_cones,
                        test_loss_d_cones/self.n_cones,
                        test_loss_x_big_cones/self.n_big_cones,
                        test_loss_y_big_cones/self.n_big_cones,
                        test_loss_d_big_cones/self.n_big_cones,
                        test_loss_d_cones_near/self.n_cones_near
                        )
        self.net.train()
    
    def save_info_model(self, 
                        test_loss, 
                        test_loss_cones, 
                        test_loss_big_cones, 
                        test_loss_x,
                        test_loss_y,
                        test_loss_d,
                        test_loss_x_cones,
                        test_loss_y_cones,
                        test_loss_d_cones,
                        test_loss_x_big_cones,
                        test_loss_y_big_cones,
                        test_loss_d_big_cones,
                        test_loss_d_cones_near
                        ):
        """
        Saves the information of the model
        """
        
        h, m, s = get_time(self.elapsed) if self.elapsed else (0, 0, 0)
        
        markdown_table = f"""
| Metric  | Value |
|-------|----|
| model name |  {self.model_name} |
| # of neurons in first layer   |  {self.net.layer1[0].out_features} |
| batch size |  {self.batch_size} |
| seed |  {self.seed} |
| lr |  {self.lr} |
| train criterion |  {self.train_criterion.__class__.__name__} |
| delta |  {self.train_criterion.delta if self.train_criterion.__class__.__name__ == "HuberLoss" else "None"} |
| test criterion |  {self.test_criterion.__class__.__name__} |
| epochs |  {self.epochs} |
| stop epoch |  {self.stop_epoch} |
| # of not inc |  {self.inc_val} |
| print every |  {self.print_every} |
| training time |  "{h} hours, {m} minutes and {s} seconds" |
| test loss |  {(test_loss)} |
| test loss small cones |  {(test_loss_cones)} |
| # of small cones |  {self.n_cones} |
| test loss big cones|  {(test_loss_big_cones)} |
| # of big cones |  {self.n_big_cones} |
| loss x |  {test_loss_x} |
| loss x small cones |  {test_loss_x_cones} |
| loss x big cones |  {test_loss_x_big_cones} |
| loss y |  {test_loss_y} |
| loss y small cones |  {test_loss_y_cones} |
| loss y big cones |  {test_loss_y_big_cones} |
| loss d |  {test_loss_d} |
| loss d small cones |  {test_loss_d_cones} |
| loss d big cones |  {test_loss_d_big_cones} |
| loss d near cones ({self.th_x}, +-{self.th_y/2})|  {test_loss_d_cones_near} |
| # of near cones |  {self.n_cones_near} |
        """
        
        self.writer_info.add_text("Results", markdown_table, 0)
        self.writer_info.flush()
        self.writer_info.close()
        
    
    def calc(self, inputs):
        """
        Returns the output of the network from a given input
        """
        
        self.load_model()
        inputs = inputs.to(self.device)
        self.net.eval()
        out = self.net(inputs)
        self.net.train()
        out = out.to("cpu")
        return out
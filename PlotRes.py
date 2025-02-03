import matplotlib.pyplot as plt
import numpy as np

def plot_train(loss, stop_epoch, path):
    epochs = np.arange(1, stop_epoch + 1) 
    loss = loss[1:stop_epoch]
    epochs = epochs[1:]
    plt.scatter(epochs, loss, color='r')
    plt.plot(epochs, loss, color='r', linestyle='-', label='training')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training")
    plt.legend()
    plt.savefig(path)
    plt.clf()

def plot_val(loss, stop_epoch, path):
    epochs = np.arange(1, stop_epoch + 1)
    loss = loss[1:stop_epoch]
    epochs = epochs[1:]
    plt.scatter(epochs, loss, color='g')
    plt.plot(epochs, loss, color='g', linestyle='-', label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation")
    plt.legend()
    plt.savefig(path)
    plt.clf()

def plot_train_val(loss_train, loss_val, stop_epoch, path):
    epochs = np.empty(stop_epoch, dtype=int)
    epochs[:] = np.arange(1, stop_epoch+1)
    epochs = epochs[1:]
    loss_train = loss_train[1:stop_epoch]
    loss_val = loss_val[1:stop_epoch]
    plt.scatter(epochs, loss_train, color='r')
    plt.plot(epochs, loss_train, color='r', linestyle='-', label='training')
    plt.scatter(epochs, loss_val, color='g')
    plt.plot(epochs, loss_val, color='g', linestyle='-', label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation")
    plt.legend()
    plt.savefig(path)
    plt.clf()
    
def plot_results(loss_train_norm, loss_val_norm, loss_train_denorm, loss_val_denorm, stop_epoch, path, norm=True):
    if norm is True:
        plot_train(loss_train_norm, stop_epoch, (path+"Train_norm.png"))
        plot_val(loss_val_norm, stop_epoch, (path+"Val_norm.png"))
        plot_train_val(loss_train_norm, loss_val_norm, stop_epoch, (path+"Train_Val_norm.png"))
    
    plot_train(loss_train_denorm, stop_epoch, (path+"Train_denorm.png"))
    plot_val(loss_val_denorm, stop_epoch, (path+"Val_denorm.png"))
    plot_train_val(loss_train_denorm, loss_val_denorm, stop_epoch, (path+"Train_Val_denorm.png"))
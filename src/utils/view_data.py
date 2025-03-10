import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.utils.tensorboard import SummaryWriter

def writeImg(x0, y0, x1, y1, x_label, y_label):
    """
    Draws images of cones distribution
    """
    
    writer = SummaryWriter("./../runs/dataset/" + x_label + "-" + y_label)
    
    # Distribution Image of big cones
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.scatter(x1, y1, c='blue', alpha=0.5)
    ax1.set_xlabel(x_label.upper())
    ax1.set_ylabel(y_label.upper())
    ax1.set_title("Big Cones " + x_label.upper() + "-" + y_label.upper())
    fig1.canvas.draw()
    image1 = np.array(fig1.canvas.renderer.buffer_rgba())
    writer.add_image("Big cones " + x_label + "-" + y_label + "/base", torch.tensor(image1).permute(2, 0, 1), global_step=0)
    plt.close(fig1)  

    # Distribution Image of small cones
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(x0, y0, c='red', alpha=0.5) 
    ax2.set_xlabel(x_label.upper())
    ax2.set_ylabel(y_label.upper())
    ax2.set_title("Small Cones " + x_label.upper() + "-" + y_label.upper())
    fig2.canvas.draw()
    image2 = np.array(fig2.canvas.renderer.buffer_rgba())
    writer.add_image("Small cones " + x_label + "-" + y_label + "/base", torch.tensor(image2).permute(2, 0, 1), global_step=0)
    plt.close(fig2)
    
    writer.flush()
    writer.close()

def writeImgHeatmap(x0, y0, x1, y1, x_label, y_label):
    """
    Draws heatmap of cones distribution
    """
    writer = SummaryWriter("./../runs/dataset/"  + x_label + "-" + y_label)

    # Heatmap for big cones
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    heatmap1 = ax1.hist2d(x1, y1, bins=50, cmap='hot', density=True, norm=colors.LogNorm())
    plt.colorbar(heatmap1[3], ax=ax1)  
    ax1.set_xlabel(x_label.upper())
    ax1.set_ylabel(y_label.upper())
    ax1.set_title("Heatmap Big Cones " + x_label.upper() + "-" + y_label.upper())
    fig1.canvas.draw()
    image1 = np.array(fig1.canvas.renderer.buffer_rgba())
    writer.add_image("Big cones " + x_label + "-" + y_label + "/heatmap", torch.tensor(image1).permute(2, 0, 1), global_step=0)
    plt.close(fig1)

    # Heatmap for small cones
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    heatmap2 = ax2.hist2d(x0, y0, bins=50, cmap='coolwarm', density=True, norm=colors.LogNorm())
    plt.colorbar(heatmap2[3], ax=ax2)
    ax2.set_xlabel(x_label.upper())
    ax2.set_ylabel(y_label.upper())
    ax2.set_title("Heatmap Small Cones " + x_label.upper() + "-" + y_label.upper())
    fig2.canvas.draw()
    image2 = np.array(fig2.canvas.renderer.buffer_rgba())
    writer.add_image("Small cones " + x_label + "-" + y_label + "/heatmap", torch.tensor(image2).permute(2, 0, 1), global_step=0)
    plt.close(fig2)
    
    writer.flush()
    writer.close()

def draw_images(x0, y0, x1, y1, x_label, y_label):
    writeImg(x0, y0, x1, y1, x_label, y_label)
    writeImgHeatmap(x0, y0, x1, y1, x_label, y_label)

def main():
    """
    Creates images and heatmap of data
    """
    
    # data loading
    file_path = "./../data/dataset/data20.dat"
    data = np.loadtxt(file_path)

    # H-W
    cones1 = data[data[:, 0] == 1][:, 3:5]
    cones0 = data[data[:, 0] == 0][:, 3:5]
    w1, h1 = cones1[:, 0], cones1[:, 1]
    w0, h0 = cones0[:, 0], cones0[:, 1]
    draw_images(w0, h0, w1, h1, "h", "w")

    # X-Y
    cones1 = data[data[:, 0] == 1][:, 5:]
    cones0 = data[data[:, 0] == 0][:, 5:]
    x1, y1 = cones1[:, 0], cones1[:, 1]
    x0, y0 = cones0[:, 0], cones0[:, 1]
    draw_images(x0, y0, x1, y1, "x", "y")

    # T-L
    cones1 = data[data[:, 0] == 1][:, 1:3]
    cones0 = data[data[:, 0] == 0][:, 1:3]
    t1, l1 = cones1[:, 0], cones1[:, 1]
    t0, l0 = cones0[:, 0], cones0[:, 1]
    draw_images(t0, l0, t1, l1, "t", "l")

    # I-J -> (centre of base side of BB)
    cones1 = data[data[:, 0] == 1][:, 1:5]
    cones0 = data[data[:, 0] == 0][:, 1:5]
    i1, j1 = cones1[:, 0] + cones1[:, 2] / 2, cones1[:, 1] + cones1[:, 3]
    i0, j0 = cones0[:, 0] + cones0[:, 2] / 2, cones0[:, 1] + cones0[:, 3]
    draw_images(i0, j0, i1, j1, "i", "j")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt

def hw(big_cones):
    x = big_cones[:, 4] # H
    y = big_cones[:, 3] # W
    plt.scatter(x, y, color='r', label='big cones')
    plt.xlabel("H")
    plt.ylabel("W")
    plt.title("HW")
    plt.legend()
    #plt.savefig("grafici/HW.png")
    plt.savefig("grafici/HW-filtered-20.png")
    plt.clf()

def ij(big_cones):
    x = np.empty(big_cones.shape[0])
    y = np.empty(big_cones.shape[0])
    for i in range(0,big_cones.shape[0]):
        x[i] = big_cones[i][1] + big_cones[i][3]/2
        y[i] = big_cones[i][2] + big_cones[i][4]
    plt.scatter(x, y, color='r', label='big cones')
    plt.xlabel("I")
    plt.ylabel("J")
    plt.title("IJ")
    plt.legend()
    #plt.savefig("grafici/IJ.png")
    plt.savefig("grafici/IJ-filtered-20.png")
    plt.clf()
    
def tl(big_cones):
    x = np.empty(big_cones.shape[0])
    y = np.empty(big_cones.shape[0])
    for i in range(0,big_cones.shape[0]):
        x[i] = big_cones[i][1]
        y[i] = big_cones[i][2]
    plt.scatter(x, y, color='r', label='big cones')
    plt.xlabel("T")
    plt.ylabel("L")
    plt.title("TL")
    plt.legend()
    #plt.savefig("grafici/TL.png")
    plt.savefig("grafici/TL-filtered-20.png")
    plt.clf()

def xy(big_cones):
    x = big_cones[:, 5] # X
    y = big_cones[:, 6] # Y
    plt.scatter(x, y, color='r', label='big cones')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY")
    plt.legend()
    #plt.savefig("grafici/XY.png") 
    plt.savefig("grafici/XY-filtered-20.png") 
    plt.clf()

def generate_graphs(big_cones):
    hw(big_cones)
    ij(big_cones)
    xy(big_cones)
    tl(big_cones)

def main():
    data = np.loadtxt("./dati/dataset/dati_grezzi.dat")
    for i in range(0,data.shape[0]):
        if (data[i][1] == 3):
            data[i][1] = 1
        else:
            data[i][1] = 0
    data_no_frame_number = data[:, 1:]
    data_filtered = data_no_frame_number[data_no_frame_number[:, 5] <= 20.0]
    big_cones = data_filtered[data_filtered[:, 0] == 1]
    np.savetxt("./dati/dataset/dati_puliti_minore_20.dat", data_filtered, fmt="%.6f")
    print(big_cones.shape[0])
    #genero i grafici utili per visualizzare distribuzione di coni grandi
    generate_graphs(big_cones)

if __name__ == "__main__":
    main()


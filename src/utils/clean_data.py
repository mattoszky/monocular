import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Cleans the data deleting instances where x coordinates > 20
    """
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

if __name__ == "__main__":
    main()


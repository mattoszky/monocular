# 📌 Cones distance estimation using monocamera 

## 📖 Description
The goal of this project is to create a neural network for estimating the distance of cones described by bounding boxes in the context of autonomous driving under SAE formula.

## 🚀 Installation
You need to have installed python3 and pip.
Follow this instruction to be able to work with the prooject:
```bash
# Clonare il repository
git clone https://github.com/tuo-username/nome-progetto.git

# Accedere alla cartella del progetto
cd monocular

# Installare le dipendenze
pip install -r requirements.txt
```

## 🔧 Usage
Now you can train your own model by setting the parameters in ```bashmain.py```.
You can choose the name of the model with the option ```bash--mn [model name]```.
For the other params you have to change them manually inside the ```bashmain.py``` file and, if you want to change the type of net, in the ```bashsolver.py``` file.

When you are ready do as follows:

```bash
# start train
python3 mian.py --mn your_model_name
```

## 🛠 Technologies
- Python
- Pytorch
- Tensorboard

## 📂 Project structure
```bash
monocular/
│-- src/                # Source Code
│   │-- nets/           # Used Neural Nets
│   │-- solvers/        # Classes to train and test nets
│   │-- utils/          # Utility files
│   │-- main.py         # Main file
│   │-- main_denorm.py  # Different version of main file
│-- data/               # all the data
│-- models/             # saved models
│-- runs/               # results
│-- README.md           # this file
```

---
✍ Made by Alessandro Mazzocchi

# 📌 Cones distance estimation using monocamera 

## 📖 Description
The goal of this project is to create a neural network for estimating the distance of cones described by bounding boxes in the context of autonomous driving under SAE formula.

## 🚀 Installation
You need to have installed python3 and pip.
Follow this instruction to be able to work with the prooject:
```bash
# Clone the repo
git clone https://github.com/mattoszky/monocular.git

# go to the project directory
cd monocular

# dependecy installation
pip install -r requirements.txt

# setup for tensorboard
export PATH=$HOME/.local/bin:$PATH
```

## 🔧 Usage
Now you can train your own model by setting the parameters in ```main.py```.
You can choose the name of the model with the option ```--mn [model name]```.
For the other params you have to change them manually inside the ```main.py``` file and, if you want to change the type of net, in the ```solver.py``` file.
You can also run a denorm version of the code with `main_denorm.py` and `solver_denorm.py`.
More information can be found inside the source code by means of comments.

When you are ready do as follows:

```bash
# start train
python3 main.py --mn your_model_name
```

After the training you can see the result with `tensorboard`. To do so go to `monocular` dir, the project directory, and run the following command:

```bash
# start tensorboard session
tensorboard --logdir=runs/
```
Then you have to click on the link and you are in, enjoy!

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
│-- data/               # All the data
│-- models/             # Saved models
│-- runs/               # Results
│-- README.md           # This file
```

---
✍ Made by Alessandro Mazzocchi

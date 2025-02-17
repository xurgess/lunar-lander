in order to run main.py for training, you must construct a conda environment

run the following commands:
- conda create --name lunar_env python=3.8
- conda activate lunar_env
- conda install gymnasium
- pip install gymnasium[box2d]
- conda install pytorch
- pip install box2d

now utilize the new lunar_env environment to run the main.py file
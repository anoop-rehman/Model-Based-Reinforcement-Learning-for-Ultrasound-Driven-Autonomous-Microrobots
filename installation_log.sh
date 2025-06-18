# ran on macos (apple silicon)

pyenv local 3.9.6
python -m venv .venv
source .venv/bin/activate
pip install pip==25.1.1

pip install numpy
pip install matplotlib
pip install scikit-image
pip install torch
pip install torchvision
pip install tensorflow_probability

pip install opencv-python
pip install opencv-contrib-python

pip install gymnasium
pip install stable-baselines3

pip install serial
pip install pyserial
pip install pyvisa
pip install pymmcore
pip install pymmcore-plus

pip install curtsies
pip install pygame

pip install git+https://github.com/Pippo809/dreamerv3.git
pip install pip==23.3.2
pip install git+https://github.com/Pippo809/dreamerv3.git
pip install tensorflow-macos
pip install --no-deps git+https://github.com/Pippo809/dreamerv3.git
pip install git+https://github.com/facebookresearch/segment-anything.git
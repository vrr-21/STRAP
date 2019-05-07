git clone https://github.com/shakenes/vizdoomgym
cd vizdoomgym/
apt-get update
apt-get install cmake libboost-all-dev libgtk2.0-dev libsdl2-dev python-numpy
pip install -e .
pip install pyvirtualdisplay
apt-get install xvfb
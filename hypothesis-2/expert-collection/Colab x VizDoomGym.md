# Steps to get VizdoomGym running on Colab

Paste this in the first cell of the notebook:

```python
!git clone https://github.com/shakenes/vizdoomgym
%cd vizdoomgym/
!apt-get update
!apt-get install cmake libboost-all-dev libgtk2.0-dev libsdl2-dev python-numpy
!pip install -e .
!pip install pyvirtualdisplay
!apt-get install xvfb
```

These will install all the dependencies required. The next lines will help checking out if the installation went fine:

```python
import gym
import vizdoomgym
from IPython.display import HTML

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display = Display(visible = 0, size = (1400,900))
display.start()

env = gym.make("VizdoomTakeCover-v0")
obs = env.reset()
print(obs.shape)
```

This should show as (240, 320, 3).
# Expert Trajectory Collection for Vizdoom games

In this section, the collection of expert trajectory for `VizdoomTakeCover` and `VizdoomDefendLine` is done. Since there were no state-of-the-art models for these games, we had to train our own 'expert' models.

We used PPO for training these expert models. The following Colab notebook shows how to use this section for creating expert models and trajectories:

[Collecting Expert Trajectories](https://colab.research.google.com/drive/1CYw7ASEMS5Sm1okt8mSM43Pqw46yco3Z)

We used the following repositories for making this code work: 
- [Stable Baselines](https://github.com/hill-a/stable-baselines): PPO code for training the expert model.
- [VizdoomGym](https://github.com/shakenes/vizdoomgym): Gym wrapper for [VizDoom](https://github.com/mwydmuch/ViZDoom).

## Requirements:
Install via pip:
- numpy
- tensorflow
- gym
- vizdoomgym
- stable-baselines

Out of these, all except vizdoomgym can be installed from the  `requirements.txt` here. For installing vizdoomgym, follow [this](https://github.com/shakenes/vizdoomgym#installation).

Install via apt-get:
- cmake
- libboost-all-dev
- libgtk2.0-dev
- libsdl2-dev
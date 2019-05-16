# Implementation of Deep Reinforcement Learning Algorithms

### Algorithms:
* [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
  * [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
  * [Prioritized Replay buffer](https://arxiv.org/pdf/1511.05952.pdf)
  * [Duelling Networks](https://arxiv.org/pdf/1511.06581.pdf)
* [A3C](https://arxiv.org/pdf/1602.01783.pdf)

### Environments
* Bipedal Walker
* Lunar Lander
* Pong
* Breakout

## Install
Create a virual environment

``` conda create -n blg604ehw2 python=3.7 ```

Activate it

``` source activate blg604ehw2 ```

Install the package at the current directory

``` pip install -e .```


<span style="color:red">**Note:**</span> If you get an error message while installing box2d:

__Unable to execute 'swig': No such file or directory__, try:

``` apt-get install swig ```

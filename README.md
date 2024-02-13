# Algorithms for Normal Form Games

This is a research repository for doing research on MARL algorithms for Normal Form Games. It is based on PyTorch.

# Installation

Clone the repository, create a venv (advised), and install the requirements:

```bash
git clone git@github.com:tboulet/Algorithms-for-Normal-Form-Games.git
cd Algorithms-for-Normal-Form-Games
python -m venv venv
source venv/bin/activate  # on linux
venv\Scripts\activate  # on windows
pip install -r requirements.txt
```


# Run the code
 
For training your algorithms on a Normal Form Game, run the following command:

```bash
python run.py algo=<algo tag> game=<game tag>
```

For example, to train the FoReL algorithm on the Matching Pennies game, run :

```bash
python run.py algo=forel game=mp
```

### Algorithms
The algo tag should correspond to a configuration in ``configs/algo/`` where you can specify the algo and its hyperparameters. 

Currently, the following algorithms are available:
 - `forel` : FoReL algorithm. At each step, it learns the Q values of each agent and each action using Monte Carlo sampling or model-based exact Q values extraction. Once the Q values are computed, it applies the FoReL rules.
 - `iforel` : Iterated FoReL Lyapunov algorithm. Apply FoReL iteratively, each iteration with the reward being modified by a regularization term that depends on the previous obtained policy.
 - `pg` : (Softmax) Policy Gradients : apply policy gradients on the policy of each agent, with the objective function being the expected advantage values.

### Games

The game tag should correspond to a configuration in ``configs/game/`` where you can specify the game and its hyperparameters.

Currently the following games are implemented :
- `mp` : Matching Pennies. In this game, there are two agents and two actions. The first agent (Even) receive +1 if the actions are the same, -1 otherwise. The second agent (Odd) receive the opposite reward. The only Nash Equilibrium (NE) that exists is (1/2, 1/2)
- `mp_bias` : a MP game biaised, where the joint action (0,0) gives rewards (4, -4) instead of (1, -1). The NE is thus slightly different.
- `mp_bias_nonzero` : here, the (0,0) returns are (4, -1), which makes the game non-zero-sum.
- `kuhn` : an NFG version of Kuhn Poker, where an action corresponds to choosing the contextual deterministic policy to be played: a ={pi(action|card) | action in A, card in S}
- `rps` : Rock-Paper-Scissors game. The NE is (1/3, 1/3, 1/3). Not currently implemented.


We use Hydra as our config system. The config folder is `./configs/`. You can modify the config (logging, metrics, number of training episodes) from the `default_config.yaml` file. You can also create your own config file and specify it with the `--config-name` argument :

```bash
python run.py algo=forel game=mp --config-name=my_config_name
```

Advice : create an alias for the command above this.


# Visualization 

The policies will be plot online during the training, every `frequency_plot` episodes, on a 2D plot. 

For 2-players 2-actions games, the x axis represent the probability of agent 0 to play action 0, and the y axis represent the probability of agent 1 to play action 0. This defines entirely the joint policy of the two agents.


# Results

### Behavior of FoReL 

Expected behavior : FoReL should PC cycles (i.e. should be Poincarré Recurrent) for any 0-sum games.

For 0-sum games :
- Using `rd` (Replicator Dynamics) as the policy update rule :
    - Using `mc` (Monte Carlo sampling) as the Q value update rule : PC cycles.
    - Using `model-based` (exact Q values extraction from the game) as the Q value update rule : FoReL cycles but slowly increase the radius of the cycle, so technically diverges.
- Using `softmax` (Softmax of the cumulative values) as the policy update rule : FoReL diverges in both cases

For non-0-sum games (e.g. `mp_bias_nonzero`) :
- Using `rd` as the policy update rule :
    - Using `mc` : cycle, but seems to reduces/increases slowly the radius of the cycle (unclear)
    - Using `model-based` : diverges slowly around a different point than the NE
- Using `softmax` as the policy update rule : diverges 
# Algorithms for Normal Form Games

This is a research repository for doing research on MARL algorithms for Normal Form Games. It is based on PyTorch.

## Installation

Clone the repository, create a venv (advised), and install the requirements:

```bash
git clone git@github.com:tboulet/Algorithms-for-Normal-Form-Games.git
cd Algorithms-for-Normal-Form-Games
python -m venv venv
source venv/bin/activate  # on linux
venv\Scripts\activate  # on windows
pip install -r requirements.txt
```


## Run the code
 
For training your algorithms on a Normal Form Game, run the following command:

```bash
python run_nfg.py algo=<algo tag> game=<game tag>
```

For example, to train the FoReL algorithm on the Matching Pennies game, run :

```bash
python run_nfg.py algo=forel game=mp
```

### Algorithms
The algo tag should correspond to a configuration in ``configs/algo/`` where you can specify the algo and its hyperparameters. 

Currently, the following algorithms are available:
 - `forel` : FoReL algorithm. At each step, it learns the Q values of each agent and each action using Monte Carlo sampling or model-based exact Q values extraction. Once the Q values are computed, it applies the FoReL rules.
 - `iforel` : Iterated FoReL Lyapunov algorithm. Apply FoReL iteratively, each iteration with the reward being modified by a regularization term that depends on the previous obtained policy.
 - `pg` : (Softmax) Policy Gradients : apply policy gradients on the policy of each agent, with the objective function being the expected advantage values.


The game tag should correspond to a configuration in ``configs/game/`` where you can specify the game and its hyperparameters.

Currentlyn the following games are implemented :


We use Hydra as our config system. The config folder is `./configs/`. You can modify the config (logging, metrics, number of training episodes) from the `default_config.yaml` file. You can also create your own config file and specify it with the `--config-name` argument :

```bash
python run_nfg.py algo=iterated_forel game=rps --config-name=my_config
```

Advice : create an alias for the command above this.

The available algorithms are in the `algorithms` sub-folder. The available games are simply in the ``configs/game/`` folder, but that may change in the future for more complex games.
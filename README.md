# Algorithms for Normal Form Games

This is a research repository for doing research on MARL algorithms for Normal Form Games. It is based on OpenSpiel and PyTorch.

## Installation

For the installation, you will need to install OpenSpiel and some stuff. You can follow the instructions in the the folder ``docs/install_openspiel.md``. You can access it byu clicking [here](docs/install_openspiel.md).


## Run the code
 
For training your algorithms on a Normal Form Game, run the following command:

```bash
python run_nfg.py algo=iterated_forel game=rps
```

The algo tag should correspond to a configuration in ``configs/algo/`` where you can specify the algo and its hyperparameters. The game tag should correspond to a configuration in ``configs/game/`` where you can specify the game and its hyperparameters.

We use Hydra as our config system. The config folder is `./configs/`. You can modify the config (logging, metrics, number of training episodes) from the `default_config.yaml` file. You can also create your own config file and specify it with the `--config-name` argument :

```bash
python run_nfg.py algo=iterated_forel game=rps --config-name=my_config
```

Advice : create an alias for the command above this.

The available algorithms are in the `algorithms` sub-folder. The available games are simply in the ``configs/game/`` folder, but that may change in the future for more complex games.
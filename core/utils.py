import sys
from typing import Any, Dict, Tuple, Union
import importlib
import numpy as np


def to_numeric(x: Union[int, float, str, None]) -> Union[int, float, None]:
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif x == "inf":
        return sys.maxsize
    elif x == "-inf":
        return -sys.maxsize
    elif x == "eps" or x == "epsilon":
        return sys.float_info.epsilon
    elif isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            raise ValueError(
                f"Cannot convert {x} to float, please specify something like '2' or '3.0' or 'inf'."
            )
    elif x is None:
        return None
    else:
        raise ValueError(f"Cannot convert {x} to numeric")


def try_get_seed(config: Dict) -> int:
    """WWill try to extract the seed from the config, or return a random one if not found

    Args:
        config (Dict): the run config

    Returns:
        int: the seed
    """
    try:
        seed = config["seed"]
        if not isinstance(seed, int):
            seed = np.random.randint(0, 1000)
    except KeyError:
        seed = np.random.randint(0, 1000)
    return seed


def try_get(dictionnary: Dict, key: str, default: Union[int, float, str, None]) -> Any:
    """Will try to extract the key from the dictionary, or return the default value if not found
    or if the value is None

    Args:
        x (Dict): the dictionary
        key (str): the key to extract
        default (Union[int, float, str, None]): the default value

    Returns:
        Any: the value of the key if found, or the default value if not found
    """
    try:
        return dictionnary[key] if dictionnary[key] is not None else default
    except KeyError:
        return default


def get_shape(
    object: Any,
    authorized_types: Tuple[type] = (np.ndarray, list, tuple, set),
) -> Tuple[int]:
    """Returns the shape of the object.
    If the object is a list, tuple, set or np.ndarray, it will return the shape of the object.
    If the object is not of the authorized types, it will return an empty tuple.

    Args:
        object (Any): the object
        authorized_types (Tuple[type], optional): the authorized types for the object. Defaults to (np.ndarray, list, tuple, set).

    Returns:
        Tuple[int]: the shape of the object
    """
    if isinstance(object, authorized_types):
        if len(object) == 0:
            return (0,)
        else:
            return (len(object), *get_shape(object[0]))
    else:
        return []


def instantiate_class(config: dict) -> Any:
    """Instantiate a class from a dictionnary that contains a key "class_string" with the format "path.to.module:ClassName"
    and that contains other keys that will be passed as arguments to the class constructor

    Args:
        config (dict): the configuration dictionnary

    Returns:
        Any: the instantiated class
    """
    assert (
        "class_string" in config
    ), "The class_string should be specified in the config"
    class_string: str = config["class_string"]
    module_name, class_name = class_string.split(":")
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class(**{k: v for k, v in config.items() if k != "class_string"})

from typing import Union


def to_numeric(x : Union[int, float, str, None]) -> Union[int, float]:
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, str):
        return float(x)
    elif x is None or x == "inf":
        return float("inf")
    elif x == "-inf":
        return float("-inf")
    else:
        raise ValueError(f"Cannot convert {x} to numeric")
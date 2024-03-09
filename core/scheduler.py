from abc import ABC, abstractmethod


class Scheduler(ABC):
    """The base class for any scheduler. A scheduler is a class that can return a value at each step."""

    def __init__(
        self,
        upper_bound: float = None,
        lower_bound: float = None,
    ):
        """Initializes the scheduler

        Args:
            upper_bound (float, optional): the upper bound of the scheduler's return. Defaults to None.
            lower_bound (float, optional): the lower bound of the scheduler's return. Defaults to None.
        """
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    @abstractmethod
    def _get_value(self, step: int) -> float:
        """The method that should be implemented by the child class to return the value at each step

        Args:
            step (int): the current step

        Returns:
            float: the value at the current step
        """
        pass

    def get_value(self, step: int) -> float:
        """Return a value at the current step. If the value is outside the bounds, it will be bounded.

        Args:
            step (int): the current step

        Returns:
            float: the value at the current step
        """
        res = self._get_value(step)
        if self.upper_bound is not None:
            assert isinstance(
                res, (int, float)
            ), f"Result returned by the scheduler is not a number : {res}, can't bound it."
            res = min(self.upper_bound, res)
        if self.lower_bound is not None:
            res = max(self.lower_bound, res)
            assert isinstance(
                res, (int, float)
            ), f"Result returned by the scheduler is not a number : {res}, can't bound it."
        return res


class Constant(Scheduler):
    """A scheduler that returns a constant value at each step."""

    def __init__(self, value: float, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _get_value(self, step: int) -> float:
        return self.value


class Linear(Scheduler):
    """A scheduler that returns a linearly increasing/decreasing value at each step."""

    def __init__(self, start_value: float, end_value: float, n_steps: int, **kwargs):
        """Initializes the Linear scheduler.
        It is parameterized by the value at the first step and the value at the n_steps-th step.
        Aftert that n-steps, the value will continue to follow the linear behavior.

        Args:
            start_value (float): the value at the first step
            end_value (float): the value at the n_steps-th step
            n_steps (int): the number of steps after which the value should be end_value.
        """
        super().__init__(**kwargs)
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def _get_value(self, step: int) -> float:
        if step > self.n_steps:
            return self.end_value
        else:
            return (
                self.start_value
                + (self.end_value - self.start_value) * step / self.n_steps
            )


class Exponential(Scheduler):
    """A scheduler that returns an exponentially increasing/decreasing value at each step."""

    def __init__(self, start_value: float, end_value: float, n_steps: int, **kwargs):
        """Initializes the Exponential scheduler.
        It is parameterized by the value at the first step and the value at the n_steps-th step.
        Aftert that n-steps, the value will continue to follow the exponential behavior.

        Args:
            start_value (float): the value at the first step
            end_value (float): the value at the n_steps-th step
            n_steps (int): the number of steps after which the value should be end_value.
        """
        super().__init__(**kwargs)
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def _get_value(self, step: int) -> float:
        return self.start_value * (self.end_value / self.start_value) ** (
            step / self.n_steps
        )


class SquareWave(Scheduler):
    """A square wave scheduler, that alternates between two values at each step."""

    def __init__(
        self,
        max_value: int,
        min_value: int,
        steps_at_min: int,
        steps_at_max: int,
        start_at_max: bool,
        **kwargs,
    ):
        """Initializes the SquareWave scheduler.
        It is parameterized by the maximum value, the minimum value, the number of steps at the minimum value and the number of steps at the maximum value.
        There is also a parameter to choose if the scheduler should start at the maximum value or not.
        
        Args:
            max_value (int): the maximum value of the square wave
            min_value (int): the minimum value of the square wave
            steps_at_min (int): the number of steps at the minimum value for each period
            steps_at_max (int): the number of steps at the maximum value for each period
            start_at_max (bool): whether to start at the maximum value or not
        """
        super().__init__(**kwargs)
        self.max_value = max_value
        self.min_value = min_value
        self.steps_at_min = steps_at_min
        self.steps_at_max = steps_at_max
        self.start_at_max = start_at_max

    def _get_value(self, step: int) -> float:
        step = step % (self.steps_at_max + self.steps_at_min)
        if self.start_at_max:
            if step < self.steps_at_max:
                return self.max_value
            else:
                return self.min_value
        else:
            if step < self.steps_at_min:
                return self.min_value
            else:
                return self.max_value

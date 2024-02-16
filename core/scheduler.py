


class Scheduler:
    
    def __init__(self, 
        type : str, 
        start_value : float,
        end_value : float,
        n_steps : int,
        upper_bound : float = float("inf"),
        lower_bound : float = float("-inf"),
        ):
        """
        Create a Scheduler object, which will be used to schedule a value over time.
        
        Args:
            - type (str): the type of scheduler (either "constant" or "linear")
            - start_value (float): the initial value
            - end_value (float): the final value
            - n_steps (int): the number of steps over which the value will change
        """
        self.type = type
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    
    def get_value(self, step : int) -> float:
        """
        Get the value of the scheduler at a given step.
        
        Args:
            - step (int): the step at which to get the value of the scheduler
        
        Returns:
            float: the value of the scheduler at the given step
        """
        if self.type == "constant":
            res = self.start_value
        elif self.type == "linear":
            if step > self.n_steps:
                res = self.end_value
            else:
                res = self.start_value + (self.end_value - self.start_value) * step / self.n_steps
        elif self.type == "exponential":
            res = self.start_value * (self.end_value / self.start_value) ** (step / self.n_steps)
        else:
            raise ValueError(f"Unknown type of scheduler : {self.type}")
        
        return min(self.upper_bound, max(self.lower_bound, res))
from mesa import Agent

class PersonAgent(Agent):
    """Represents a bus moving in the canyon."""
    def __init__(self, model):
        super().__init__(model)
        self.position = position  # The index of the segment
        self.status = 'im a person'
        self.car_preference = .5
        self.time_preference = 0 

    def adjust_speed(self):
        """Tracks occupancy but does not move."""
        pass
        
    def move_along_path(self):
        pass

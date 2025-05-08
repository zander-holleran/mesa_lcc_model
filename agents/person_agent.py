from mesa import Agent

class PersonAgent(Agent):
    """Represents a bus moving in the canyon."""
    def __init__(self, model):
        super().__init__(model)
        self.position = position  # The index of the segment
        self.status = 'im a person'

    def adjust_speed(self):
        """Tracks occupancy but does not move."""
        pass
        
    def move_along_path(self):
        pass

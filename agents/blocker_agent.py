from mesa import Agent

class BlockerAgent(Agent):
    """Represents a segment of the road. Only one car can occupy it at a time."""
    
    def __init__(self, model, road_segment, blocker_type, self_distruct_timer):
        allowed_types = {"crash", "canyon_closure"}
        if blocker_type not in allowed_types:
            raise ValueError(f"blocker_type must be one of {allowed_types}, got '{blocker_type}'")
        
        super().__init__(model)
        
        # stuff to do with the road segment where the block will take place
        self.position = road_segment.position  # The index of the segment
        self.distance_traveled = road_segment.distance_traveled

        self.self_distruct_timer = self_distruct_timer
        self.status = blocker_type # can be crash or canyon_closure


       
    
    def self_distruct(self):
        # Remove from the model's AgentSet
        print(f"Blocker at position {self.position} self-destructed.")
        self.model.agents.remove(self)

    def tick(self):
        self.self_distruct_timer -= 1
        if self.self_distruct_timer <= 0:
            self.self_distruct()


    # stuff so that the step function does not break
    def get_next_agent(self):
        pass

    def adjust_speed(self):
        pass
        
    def move_along_path(self):
        pass

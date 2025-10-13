from mesa import Agent
from agents.vehicle_agent import VehicleAgent

class BlockerAgent(Agent):
    """Represents a segment of the road. Only one car can occupy it at a time."""
    
    def __init__(self, model, road_segment, blocker_type, self_distruct_timer):
        allowed_types = {"crash", "canyon_closure"}
        if blocker_type not in allowed_types:
            raise ValueError(f"blocker_type must be one of {allowed_types}, got '{blocker_type}'")
        
        super().__init__(model)
        # stuff to do with the road segment where the block will take place
        self.self_distruct_timer = self_distruct_timer
        self.status = blocker_type # can be crash or canyon_closure
        # road_segment stuff
        self.pos = road_segment.pos  # The index of the segment
        self.distance_traveled = road_segment.distance_traveled
        self.speed=0

        # IMPORTANT upon blocker creation nxt agent of all vehicles is reset to None
        self.model.agents.select(agent_type=VehicleAgent).do("reset_next_agent") 
        
    def self_distruct(self):
        # Remove from the model's AgentSet
        print(f"Blocker {self.unique_id} self-destructed.")
        self.model.agents.select(agent_type=VehicleAgent).do("reset_next_agent") 
        self.model.agents.remove(self)

    def tick(self):
        self.self_distruct_timer -= 1
        if self.self_distruct_timer <= 0:
            self.self_distruct()


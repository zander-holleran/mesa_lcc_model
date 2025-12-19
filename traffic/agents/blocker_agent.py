from mesa import Agent
from traffic.agents.vehicle_agent import VehicleAgent

class BlockerAgent(Agent):
    """Represents a segment of the road. Only one car can occupy it at a time."""
    
    def __init__(self, model, seg_i, blocker_type, self_distruct_timer):
        allowed_types = {"crash", "canyon_closure"}
        if blocker_type not in allowed_types:
            raise ValueError(f"blocker_type must be one of {allowed_types}, got '{blocker_type}'")
        
        super().__init__(model)

        print(f'you made a blocker: {blocker_type}')

        # stuff to do with the road segment where the block will take place
        self.self_distruct_timer = self_distruct_timer
        self.status = blocker_type # can be crash or canyon_closure
        # road_segment stuff
        self.pos = self.model.rs_pos[seg_i]  # The index of the segment
        self.distance_traveled = self.model.rs_distance[seg_i]
        self.speed=0

        # IMPORTANT upon blocker creation nxt agent of all vehicles is reset to None
        self.model.agents.select(agent_type=VehicleAgent).do("reset_next_agent") 

        # holder to speed up next agent assignment
        self.next_agent = None
        self.gap = 9_999_999.0  # distance to next vehicle (m) starts
        
    def self_distruct(self):
        # Remove from the model's AgentSet
        self.model.agents.select(agent_type=VehicleAgent).do("reset_next_agent") # IMPORTANT upon blocker destruction nxt agent of all vehicles is reset to None
        try: self.model.blockers_list.remove(self)   # drop from vehicles list
        except ValueError: pass
        self.model.agents.remove(self)               # drop from agent set

    def tick(self):
        self.self_distruct_timer -= 1
        if self.self_distruct_timer <= 0:
            self.self_distruct()


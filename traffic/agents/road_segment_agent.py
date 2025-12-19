from mesa import Agent

class RoadSegmentAgent(Agent):
    """Represents a segment of the road. Only one car can occupy it at a time."""
    
    def __init__(self, model, speed_limit, curvature, linked_coord, road_section, distance_traveled):
        super().__init__(model)
        #self.position = position  # The index of the segment
        self.occupied = False  # Whether a car is on this segment
        self.status = 'im just a road'
        self.speed_limit = speed_limit
        self.road_section = road_section
        self.curvature = curvature
        self.linked_coord = linked_coord
        self.distance_traveled = distance_traveled
    
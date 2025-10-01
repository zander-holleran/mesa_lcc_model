from agents.road_segment_agent import RoadSegmentAgent
 
def init_road_segments(model, road_gdf, canyon_open_step=None, closed_sections=None):
    closed_sections = closed_sections or set()

    road_closed = [
        True if (canyon_open_step is not None and section in closed_sections) else False
        for section in road_gdf.road_section
    ]
    
    return RoadSegmentAgent.create_agents(
        model=model,
        n=len(road_gdf),
        position=[(pt.x, pt.y) for pt in road_gdf.geometry],
        speed_limit=road_gdf.speed_limit.tolist(),
        road_section=road_gdf.road_section.tolist(),
        curvature=road_gdf.curvature.tolist(),
        road_closed = road_closed,
        linked_coord=road_gdf.linked_coord.tolist(),
        distance_traveled=road_gdf.distance_traveled.tolist()
    )
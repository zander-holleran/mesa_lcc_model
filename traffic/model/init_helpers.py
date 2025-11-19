from traffic.agents.road_segment_agent import RoadSegmentAgent
 
def init_road_segments(model, road_gdf):

    return RoadSegmentAgent.create_agents(
        model=model,
        n=len(road_gdf),
        #position=[(pt.x, pt.y) for pt in road_gdf.geometry],
        speed_limit=road_gdf.speed_limit.tolist(),
        road_section=road_gdf.road_section.tolist(),
        curvature=road_gdf.curvature.tolist(),
        linked_coord=road_gdf.linked_coord.tolist(),
        distance_traveled=road_gdf.distance_traveled.tolist()
    )
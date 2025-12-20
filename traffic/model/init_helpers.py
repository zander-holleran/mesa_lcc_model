from traffic.agents.road_segment_agent import RoadSegmentAgent


def init_road_segments(model, road_gdf):
    # store fast arrays for hot-path lookups (no AgentSet indexing)
    model.rs_pos = [(pt.x, pt.y) for pt in road_gdf.geometry]
    model.rs_distance = road_gdf.distance_traveled.to_numpy()
    model.rs_speed_limit = road_gdf.speed_limit.to_numpy()
    model.rs_road_section = road_gdf.road_section.to_numpy()
    model.rs_curvature = road_gdf.curvature.to_numpy()
    model.rs_linked_coord = road_gdf.linked_coord.tolist()

    # keep creating the agents if you still need them for other stuff
    return RoadSegmentAgent.create_agents(
        model=model,
        n=len(road_gdf),
        speed_limit=road_gdf.speed_limit.tolist(),
        road_section=road_gdf.road_section.tolist(),
        curvature=road_gdf.curvature.tolist(),
        linked_coord=road_gdf.linked_coord.tolist(),
        distance_traveled=road_gdf.distance_traveled.tolist(),
    )
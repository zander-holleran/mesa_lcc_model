import numpy as np
from traffic.agents.road_segment_agent import RoadSegmentAgent


def init_road_segments(model, road_gdf):
    # store fast arrays for hot-path lookups (no AgentSet indexing)
    model.rs_pos = [(pt.x, pt.y) for pt in road_gdf.geometry]
    model.rs_distance = road_gdf.distance_traveled.to_numpy()
    model.rs_speed_limit = road_gdf.speed_limit.to_numpy()
    model.rs_road_section = road_gdf.road_section.to_numpy()
    model.rs_curvature = road_gdf.curvature.to_numpy()
    model.rs_linked_coord = road_gdf.linked_coord.tolist()

    # Upgrade rs_pos from list-of-tuples to numpy array, then apply start_point override
    model.rs_pos = np.array([(pt.x, pt.y) for pt in road_gdf.geometry], dtype=np.float64)  # shape (N, 2)
    model.rs_pos[0] = np.array(model.start_point, dtype=np.float64)   # match per-vehicle path_xy[0] = p_start

    # Segment geometry (N-1 segments for N points)
    seg_vecs = model.rs_pos[1:] - model.rs_pos[:-1]                   # shape (N-1, 2)
    rs_seg_len = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])             # shape (N-1,)
    rs_seg_len[rs_seg_len == 0.0] = 1e-12                              # guard zero-length segments
    model.rs_seg_len = rs_seg_len                                      # shape (N-1,)
    model.rs_seg_dir = seg_vecs / rs_seg_len[:, np.newaxis]            # shape (N-1, 2)
    model.rs_cumulative_s = np.cumsum(rs_seg_len)                      # shape (N-1,), road total = rs_cumulative_s[-1]

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
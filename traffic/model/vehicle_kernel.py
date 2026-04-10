import numpy as np

STATUS_DRIVING  = np.int8(0)
STATUS_SLOWING  = np.int8(1)
STATUS_CRASH    = np.int8(2)
STATUS_CC       = np.int8(3)

STATUS_MAP = {'driving': STATUS_DRIVING, 'slowing': STATUS_SLOWING,
              'crash': STATUS_CRASH, 'canyon_closure': STATUS_CC}

ACTION_ACCELERATE    = np.int8(0)
ACTION_COAST         = np.int8(1)
ACTION_SLOW_ACCEL    = np.int8(2)
ACTION_SL_BRAKE      = np.int8(3)
ACTION_SMOOTH_BRAKE  = np.int8(4)
ACTION_PREVENT_PASS  = np.int8(5)

MPS_TO_MPH = 2.23694
MPH_TO_MPS = 1.0 / MPS_TO_MPH


def step(model) -> list:
    vs = model.vs
    n = vs.n_active
    if n == 0:
        return []

    # ── 1. Build combined vehicle + blocker arrays for ordering ──────────────
    blockers = model.blockers_list
    n_b = len(blockers)

    veh_dist   = vs.dist[:n].copy()    # copy so sort doesn't affect store until writeback
    veh_speed  = vs.speed[:n].copy()
    veh_status = vs.status[:n].copy()

    if n_b > 0:
        blk_dist   = np.array([b.distance_traveled for b in blockers], dtype=np.float64)
        blk_status = np.array([STATUS_MAP[b.status] for b in blockers], dtype=np.int8)
        blk_speed  = np.zeros(n_b, dtype=np.float64)
        comb_dist   = np.concatenate([veh_dist,   blk_dist])
        comb_speed  = np.concatenate([veh_speed,  blk_speed])
        comb_status = np.concatenate([veh_status, blk_status])
        is_vehicle  = np.array([True] * n + [False] * n_b)
    else:
        comb_dist   = veh_dist
        comb_speed  = veh_speed
        comb_status = veh_status
        is_vehicle  = np.ones(n, dtype=bool)

    n_comb = n + n_b

    # ── 2. Sort combined by distance ─────────────────────────────────────────
    comb_ord = np.argsort(comb_dist, kind='stable')
    sorted_dist   = comb_dist[comb_ord]
    sorted_status = comb_status[comb_ord]
    sorted_speed  = comb_speed[comb_ord]
    sorted_is_veh = is_vehicle[comb_ord]

    # Inverse map: original index → rank in sorted order
    inv_ord = np.empty(n_comb, dtype=np.int32)
    inv_ord[comb_ord] = np.arange(n_comb, dtype=np.int32)
    veh_ranks = inv_ord[:n]   # rank of each vehicle slot in sorted combined order

    # ── 3. Gap to next entity (vehicle or blocker) ───────────────────────────
    has_leader = (veh_ranks + 1) < n_comb
    leader_rank = np.minimum(veh_ranks + 1, n_comb - 1)
    gap = sorted_dist[leader_rank] - vs.dist[:n]
    gap = np.maximum(gap, 0.0)
    gap[~has_leader] = np.inf  # frontmost vehicles have no leader
    vs.gap[:n] = gap

    leader_status = sorted_status[leader_rank]
    leader_speed  = sorted_speed[leader_rank]
    # Frontmost vehicles: treat leader as driving at high speed
    leader_status[~has_leader] = STATUS_DRIVING
    leader_speed[~has_leader] = 1e6

    # ── 4. Adjust status ─────────────────────────────────────────────────────
    ideal_gap = vs.ideal_gap[:n]
    # slowing: gap <= ideal_gap AND leader is not freely driving
    slowing_mask = (gap <= ideal_gap) & (leader_status != STATUS_DRIVING)
    new_status = np.where(slowing_mask, STATUS_SLOWING, STATUS_DRIVING)

    # Backward cascade: inherit crash/cc status from a stopped blocker when very slow
    nearly_stopped = vs.speed[:n] < 1.0   # m/s
    inherit_mask = nearly_stopped & (
        (leader_status == STATUS_CRASH) | (leader_status == STATUS_CC)
    )
    new_status[inherit_mask] = leader_status[inherit_mask]
    vs.status[:n] = new_status

    # ── 5. Filter to active movers ───────────────────────────────────────────
    active = (new_status == STATUS_DRIVING) | (new_status == STATUS_SLOWING)

    # ── 6. Speed limit lookup (weighted 5-segment lookahead) ─────────────────
    N_ahead = 5
    n_segs = len(model.rs_speed_limit)
    path_idx = vs.path_idx[:n]

    # Build (n, N_ahead) index array clamped to valid segment range
    offsets = np.arange(N_ahead, dtype=np.int32)
    lookahead_idx = np.minimum(path_idx[:, np.newaxis] + offsets, n_segs - 1)  # (n, N_ahead)

    sl_ahead   = model.rs_speed_limit[lookahead_idx]   # (n, N_ahead), mph (from UDOT)
    curv_ahead = model.rs_curvature[lookahead_idx]     # (n, N_ahead)

    # Weights: [1, 0.5, 0.333, 0.25, 0.2], normalized (same as VehicleAgent._weights)
    raw_w = np.array([1.0 / (1 + i) for i in range(N_ahead)], dtype=np.float64)
    w = raw_w / raw_w.sum()

    avg_sl_mph  = sl_ahead  @ w   # (n,) — already in mph
    avg_curv    = curv_ahead @ w  # (n,)

    acceptable_over = vs.acceptable_ov[:n].astype(np.float64)
    curve_resp = vs.curve_resp[:n].astype(np.float64)

    # Curve adjustment (mirrors VehicleAgent.get_speed_limit)
    curve_effect = np.clip(avg_curv / 90.0, 0.0, 1.0)
    speed_effect = np.clip((avg_sl_mph - 10.0) / (60.0 - 10.0), 0.0, 1.0)
    speed_effect[avg_sl_mph <= 15.0] = 0.0
    curve_sl_mph = avg_sl_mph * (1.0 - curve_resp * curve_effect * speed_effect)
    implicit_sl_mph = curve_sl_mph + acceptable_over
    implicit_sl_mps = implicit_sl_mph * MPH_TO_MPS

    vs.implicit_sl[:n] = implicit_sl_mps

    # ── 7. Speed update (priority branches via masks) ─────────────────────────
    speed = vs.speed[:n].copy()
    break_cd = vs.break_cd[:n].copy()
    action = vs.driving_action[:n].copy()
    idm = vs.ideal_dm[:n].astype(np.float64)

    # Masks (mutually exclusive priority):
    gap_brake_mask  = active & (gap < np.maximum(speed * idm, 5.0))
    sl_brake_mask   = active & ~gap_brake_mask & (speed > implicit_sl_mps + 1e-6)
    accel_mask      = active & ~gap_brake_mask & ~sl_brake_mask

    # --- Gap brake ---
    force = np.clip((np.maximum(speed * idm, 5.0) - gap) / np.maximum(np.maximum(speed * idm, 5.0), 1e-6), 0.0, 1.0)
    noise = model.rng.normal(0.0, 0.1, n)
    decel = force ** 2 * 8.0 + noise
    speed[gap_brake_mask] -= decel[gap_brake_mask]
    break_cd[gap_brake_mask] = 5
    action[gap_brake_mask] = ACTION_SMOOTH_BRAKE
    vs.car_interactions[:n] += gap_brake_mask.astype(np.int32)

    # --- Speed limit brake ---
    over_mph = (speed - implicit_sl_mps) * MPS_TO_MPH
    # mirrors speed_limit_brake: piecewise decel based on how far over
    sl_decel_mps = np.where(
        over_mph < 5.0,
        over_mph * MPH_TO_MPS * 0.3,
        over_mph * MPH_TO_MPS * 0.6,
    )
    speed[sl_brake_mask] -= sl_decel_mps[sl_brake_mask]
    break_cd[sl_brake_mask] = 3
    action[sl_brake_mask] = ACTION_SL_BRAKE

    # --- Accelerate (with cooldown ramp) ---
    # Acceleration magnitude from accel_curve per performance group
    speed_mph = speed * MPS_TO_MPH
    perf_group = (vs.performance[:n] * 100).astype(np.int32)
    perf_group = np.clip(perf_group, 0, 100)

    accel_delta = np.zeros(n, dtype=np.float64)
    for g in np.unique(perf_group[accel_mask]):
        grp_mask = accel_mask & (perf_group == g)
        if grp_mask.any():
            accel_fn = model.accel_curve_cache[g]
            vec_fn = np.vectorize(accel_fn)
            accel_delta[grp_mask] = vec_fn(speed_mph[grp_mask])

    coast_mask      = accel_mask & (break_cd >= 4)
    slow_accel_mask = accel_mask & (break_cd >= 1) & (break_cd < 4)
    full_accel_mask = accel_mask & (break_cd == 0)

    action[coast_mask]      = ACTION_COAST
    action[slow_accel_mask] = ACTION_SLOW_ACCEL
    action[full_accel_mask] = ACTION_ACCELERATE

    ramp = np.where(slow_accel_mask, (4 - break_cd) / 4.0, 1.0)
    speed[slow_accel_mask] += accel_delta[slow_accel_mask] * ramp[slow_accel_mask]
    speed[full_accel_mask] += accel_delta[full_accel_mask]

    break_cd[coast_mask]      = np.maximum(break_cd[coast_mask] - 1, 0)
    break_cd[slow_accel_mask] = np.maximum(break_cd[slow_accel_mask] - 1, 0)

    speed = np.maximum(speed, 0.0)

    # --- Prevent-pass backward clamp ---
    sorted_veh_mask = sorted_is_veh   # True where sorted position is a vehicle
    sorted_veh_slots = comb_ord[:n_comb][sorted_veh_mask]  # original slot indices, front to back

    for rank_idx in range(len(sorted_veh_slots) - 1, -1, -1):
        slot_i = sorted_veh_slots[rank_idx]
        if slot_i >= n or not active[slot_i]:
            continue
        leader_combined_rank = veh_ranks[slot_i] + 1
        if leader_combined_rank >= n_comb:
            continue
        if not sorted_is_veh[leader_combined_rank]:
            continue   # leader is a blocker; no prevent-pass
        leader_slot = comb_ord[leader_combined_rank]
        if leader_slot >= n:
            continue
        would_pass = (speed[slot_i] - speed[leader_slot]) > gap[slot_i]
        if would_pass:
            speed[slot_i] = speed[leader_slot] - 1.0
            break_cd[slot_i] = 5
            action[slot_i] = ACTION_PREVENT_PASS

    # Restore pre-refactor speed floor: vehicles must never move backward
    speed = np.maximum(speed, 0.0)

    # Write speed/cooldown/action back to store
    vs.speed[:n]          = speed
    vs.break_cd[:n]       = break_cd
    vs.driving_action[:n] = action

    # Update ideal_gap post speed update
    vs.ideal_gap[:n] = np.maximum(speed * idm, 5.0)

    # ── 8. Movement ──────────────────────────────────────────────────────────
    new_dist = vs.dist[:n] + speed   # continuous float, meters from road start

    rs_cs = model.rs_cumulative_s    # shape (N-1,)
    n_segs_1 = len(rs_cs)            # N-1

    new_path_idx = np.searchsorted(rs_cs, new_dist, side='right')
    new_path_idx = np.minimum(new_path_idx, n_segs_1 - 1)

    prev_cum = np.where(new_path_idx > 0, rs_cs[new_path_idx - 1], 0.0)
    new_seg_s = new_dist - prev_cum

    new_pos_xy = (
        model.rs_pos[new_path_idx]
        + model.rs_seg_dir[new_path_idx] * new_seg_s[:, np.newaxis]
    )

    arrived_mask = new_dist >= vs.route_end_dist[:n]
    # Clamp arrived vehicles to road end
    new_pos_xy[arrived_mask] = model.rs_pos[-1]
    new_path_idx[arrived_mask] = n_segs_1 - 1
    new_seg_s[arrived_mask] = rs_cs[-1] - (rs_cs[-2] if n_segs_1 > 1 else 0.0)

    vs.dist[:n]     = new_dist
    vs.path_idx[:n] = new_path_idx
    vs.seg_s[:n]    = new_seg_s
    vs.pos_x[:n]    = new_pos_xy[:, 0]
    vs.pos_y[:n]    = new_pos_xy[:, 1]
    vs.steps_taken[:n] += 1

    # ── 9. Time lost ─────────────────────────────────────────────────────────
    speed_mph_final = speed * MPS_TO_MPH
    implicit_sl_mph_final = implicit_sl_mps * MPS_TO_MPH
    speed_delta = speed_mph_final - implicit_sl_mph_final
    time_lost = np.maximum(-speed_delta / np.maximum(implicit_sl_mph_final, 1e-6), 0.0)
    vs.speed_delta[:n]   = speed_delta
    vs.cumtime_lost[:n] += time_lost

    # ── 10. Return arrived vehicle shells for end_of_road() ──────────────────
    arrived_vids = vs.slot_to_vid[:n][arrived_mask]
    return [model.vid_to_vehicle[int(vid)] for vid in arrived_vids]

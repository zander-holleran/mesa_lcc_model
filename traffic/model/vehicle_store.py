import numpy as np


class VehicleStore:
    """
    Persistent numpy arrays for all active vehicle dynamic state.
    Arrays are pre-allocated to max_vehicles. Active vehicles occupy slots [:n_active].
    Slot management uses swap-with-tail for O(1) removal.
    """

    def __init__(self, max_vehicles: int = 1500):
        M = max_vehicles
        self.MAX = M
        self.n_active = 0

        # Slot ↔ vid mappings
        self.vid_to_slot: dict[int, int] = {}
        self.slot_to_vid = np.zeros(M, dtype=np.int64)

        # --- Dynamic state (updated every step by kernel) ---
        self.dist             = np.zeros(M, dtype=np.float64)   # meters from road start
        self.path_idx         = np.zeros(M, dtype=np.int32)     # current segment index
        self.seg_s            = np.zeros(M, dtype=np.float64)   # meters into current segment
        self.speed            = np.zeros(M, dtype=np.float64)   # m/s
        self.break_cd         = np.zeros(M, dtype=np.int8)      # cooldown counter 5→0
        self.status           = np.zeros(M, dtype=np.int8)      # 0=driving 1=slowing 2=crash 3=cc
        self.gap              = np.full(M, np.inf, dtype=np.float64)
        self.ideal_gap        = np.zeros(M, dtype=np.float64)
        self.implicit_sl      = np.zeros(M, dtype=np.float64)   # m/s
        self.speed_delta      = np.zeros(M, dtype=np.float64)   # mph over implicit limit
        self.cumtime_lost     = np.zeros(M, dtype=np.float64)   # cumulative seconds lost
        self.car_interactions = np.zeros(M, dtype=np.int32)
        self.steps_taken      = np.zeros(M, dtype=np.int32)
        self.driving_action   = np.zeros(M, dtype=np.int8)      # encoded, see ACTION_MAP in kernel
        self.pos_x            = np.zeros(M, dtype=np.float64)   # 2D position (replaces Mesa space)
        self.pos_y            = np.zeros(M, dtype=np.float64)

        # --- Immutable traits (written at spawn, never changed) ---
        self.veh_type         = np.zeros(M, dtype=np.int8)      # 0=car 1=bus
        self.acceptable_ov    = np.zeros(M, dtype=np.float32)
        self.ideal_dm         = np.zeros(M, dtype=np.float32)
        self.curve_resp       = np.zeros(M, dtype=np.float32)
        self.performance      = np.zeros(M, dtype=np.float32)
        self.route_end_dist   = np.zeros(M, dtype=np.float64)   # rs_cumulative_s[-1] for all vehicles
        self.created_step     = np.zeros(M, dtype=np.int32)
        self.toll_paid        = np.zeros(M, dtype=np.float32)

    def add(self, vid: int, veh_type: int, pos_x: float, pos_y: float,
            speed: float, route_end_dist: float,
            acceptable_ov: float, ideal_dm: float, curve_resp: float,
            performance: float, created_step: int, toll_paid: float) -> int:
        """
        Write spawn-time values into the next free slot.
        Dynamic fields (dist, path_idx, seg_s, gap, etc.) start at their zero-initialized values.
        Returns the slot index.
        """
        if self.n_active >= self.MAX:
            raise RuntimeError(
                f"VehicleStore full ({self.MAX}). Increase max_concurrent_vehicles."
            )
        slot = self.n_active
        self.slot_to_vid[slot] = vid
        self.vid_to_slot[vid] = slot

        self.veh_type[slot]      = veh_type
        self.pos_x[slot]         = pos_x
        self.pos_y[slot]         = pos_y
        self.speed[slot]         = speed
        self.route_end_dist[slot]= route_end_dist
        self.acceptable_ov[slot] = acceptable_ov
        self.ideal_dm[slot]      = ideal_dm
        self.curve_resp[slot]    = curve_resp
        self.performance[slot]   = performance
        self.created_step[slot]  = created_step
        self.toll_paid[slot]     = toll_paid
        self.gap[slot]           = np.inf
        self.status[slot]        = 0  # driving
        self.break_cd[slot]      = 0
        self.dist[slot]          = 0.0
        self.path_idx[slot]      = 0
        self.seg_s[slot]         = 0.0
        self.cumtime_lost[slot]  = 0.0
        self.car_interactions[slot] = 0
        self.steps_taken[slot]   = 0
        self.implicit_sl[slot]   = 0.0
        self.speed_delta[slot]   = 0.0
        self.ideal_gap[slot]     = 0.0
        self.driving_action[slot] = 0

        self.n_active += 1
        return slot

    def remove(self, vid: int) -> None:
        """
        Swap-with-tail removal. The last active slot is copied into the freed slot.
        Updates vid_to_slot and slot_to_vid mappings.
        """
        slot = self.vid_to_slot.pop(vid)
        last = self.n_active - 1

        if slot != last:
            # Copy last slot into freed slot for every array
            moved_vid = int(self.slot_to_vid[last])
            for arr in self._all_dynamic + self._all_traits:
                arr[slot] = arr[last]
            self.slot_to_vid[slot] = moved_vid
            self.vid_to_slot[moved_vid] = slot

        self.n_active -= 1

    @property
    def _all_dynamic(self):
        return [
            self.dist, self.path_idx, self.seg_s, self.speed, self.break_cd,
            self.status, self.gap, self.ideal_gap, self.implicit_sl, self.speed_delta,
            self.cumtime_lost, self.car_interactions, self.steps_taken,
            self.driving_action, self.pos_x, self.pos_y,
        ]

    @property
    def _all_traits(self):
        return [
            self.veh_type, self.acceptable_ov, self.ideal_dm, self.curve_resp,
            self.performance, self.route_end_dist, self.created_step, self.toll_paid,
        ]

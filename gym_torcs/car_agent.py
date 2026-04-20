"""
car_agent.py  -  Alpha driver (v0.2)
Goal: drive forward, stay on track, handle sharp chicanes without crashing.
Rule-based control with look-ahead corner detection via track sensors.

Key upgrades over v0.1:
  - Adaptive speed target based on forward track sensor readings
  - Preemptive braking before corners (no more sailing into chicanes at 150)
  - Steering gain scales with speed (more aggressive at low speed)
  - Smarter traction control with wheelspin detection
  - Chicane handling: detects sharp left-right-left sequences via sensor asymmetry
"""

import numpy as np

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
TARGET_SPEED      = 290      # km/h  - cruise target on open straights

MAX_ACCEL         = 1.0      # full throttle available

# Steering gains
STEER_ANGLE_GAIN  = 12.0     # was 22.0 — main oscillation source
STEER_CENTER_GAIN = 0.8      # was 1.6
STEER_LOOKAHEAD   = 0.3      # was 0.9 — barely blend it in

# Adaptive speed table — forward_dist = min of sensors 7-11 (±30° cone)
# Tuned for F1 intent: hold speed on straights, commit hard at corner entry
LOOKAHEAD_SPEEDS = [
    (12,   40),   # wall imminent  → emergency
    (25,   70),   # hairpin        → slow
    (45,  115),   # tight corner   → was 95,  apex helps here
    (70,  160),   # medium corner  → was 135, main beneficiary
    (110, 210),   # gentle curve   → was 190
    (150, TARGET_SPEED),
]

# Corner severity modifier (mid-sensor asymmetry)
ASYM_TIGHT_THRESH  = 0.45    # above this → cap at 80 km/h (was 0.5 → 75)
ASYM_MED_THRESH    = 0.25    # above this → cap at 105 km/h (was 0.3 → 95)
ASYM_TIGHT_SPEED   = 90.0
ASYM_MED_SPEED     = 120.0
APEX_OFFSET       = 0.65    # how far toward inside wall (0=centre, 1=edge)
APEX_ASYM_THRESH  = 0.12    # min asymmetry to trigger apex mode (filters straights)

# Braking
BRAKE_GAIN        = 0.12     # per km/h over adaptive target   (was 0.06)
MAX_BRAKE         = 1.00     # full F1 braking                 (was 0.80)
BRAKE_TRIGGER     = 2        # km/h over target before braking (was 5)

# Push-to-pass: if ALL forward sensors exceed this, we're on a straight
PTP_DIST_THRESHOLD = 160     # metres clear ahead → full throttle

# F1-style gear map — hold gears much longer for acceleration
#    idx:    1    2    3    4    5    6
GEAR_UP   = [0,  55, 100, 145, 190, 240]   # 1st→2nd at 55 (was 85)
GEAR_DOWN = [0,   0,  40,  80, 120, 165]   # downshift thresholds

# Traction control — loosened to allow more wheelspin on exit
TC_THRESHOLD      = 10.0     # was 5.0 — allow a bit of controlled spin
TC_REDUCTION      = 0.15     # was 0.20

# Track boundary emergency: if |trackPos| exceeds this → hard steer back
TRACK_BOUNDARY    = 0.82


class AlphaDriver:

    def __init__(self):
        self.gear = 1
        self._last_steer = 0.0

    # ------------------------------------------------------------------
    # Primary interface — gym_torcs namedtuple
    # ------------------------------------------------------------------
    def act(self, obs, reward=0.0, done=False, vision=False):
        track_arr = np.array(obs.track, dtype=np.float32)
        steer = self._compute_steer_from_obs(obs, track_arr)
        return np.array([steer], dtype=np.float32)

    # ------------------------------------------------------------------
    # Raw interface — snakeoil ServerState / DriverAction dicts
    # ------------------------------------------------------------------
    def act_raw(self, S, R):
        s = S.d
        r = R.d

        track     = np.array(s['track'], dtype=np.float32)
        speed     = float(s['speedX'])
        angle     = float(s['angle'])
        track_pos = float(s['trackPos'])

        # ----------------------------------------------------------------
        # 1.  Adaptive speed target
        # ----------------------------------------------------------------
        adaptive_target = self._adaptive_target(track, speed)

        # ----------------------------------------------------------------
        # 2.  Steering
        # ----------------------------------------------------------------
        steer = self._compute_steer_raw(angle, track_pos, track, speed)
        r['steer'] = steer

        # ----------------------------------------------------------------
        # 3.  Throttle / Brake
        # ----------------------------------------------------------------
        speed_err = speed - adaptive_target

        # Push-to-pass: straight line detected → max throttle
        forward_clear = float(np.min(track[7:12])) > PTP_DIST_THRESHOLD

        if speed_err < -BRAKE_TRIGGER:
            gap = adaptive_target - speed
            if forward_clear:
                # Straight — full attack
                accel = MAX_ACCEL
            else:
                # Corner approach — proportional ramp-up
                accel = float(np.clip(0.15 + gap / (adaptive_target + 1.0),
                                      0.0, MAX_ACCEL))
                # Smooth approach within 25 km/h of target
                if speed_err > -25:
                    accel *= 0.65

            r['accel'] = accel
            r['brake'] = 0.0

        elif speed_err > BRAKE_TRIGGER:
            # Over target → brake hard (F1 style)
            r['accel'] = 0.0
            r['brake'] = float(np.clip(speed_err * BRAKE_GAIN,
                                       0.0, MAX_BRAKE))
        else:
            # Sweet spot — light cruise
            r['accel'] = 0.08
            r['brake'] = 0.0

        # Standstill kick
        if speed < 5:
            r['accel'] = min(1.0, r['accel'] + 1.0 / (speed + 0.5))
            r['brake'] = 0.0

        # ----------------------------------------------------------------
        # 4.  Traction control
        # ----------------------------------------------------------------
        front = s['wheelSpinVel'][0] + s['wheelSpinVel'][1]
        rear  = s['wheelSpinVel'][2] + s['wheelSpinVel'][3]
        if rear - front > TC_THRESHOLD:
            r['accel'] = max(0.0, r['accel'] - TC_REDUCTION)

        # ----------------------------------------------------------------
        # 5.  Gear
        # ----------------------------------------------------------------
        current_gear = int(s.get('gear', self.gear)) or 1
        r['gear'] = self._auto_gear(speed, int(max(1, current_gear)))
        self.gear = r['gear']

        # ----------------------------------------------------------------
        # 6.  Housekeeping
        # ----------------------------------------------------------------
        r['clutch'] = 0.0
        r['meta']   = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_target(self, track: np.ndarray, speed: float) -> float:
        """
        Speed target from track sensors.
        forward_dist — how open the road is directly ahead (±30° cone)
        mid_asymmetry — left/right imbalance flags a corner
        """
        forward_dist  = float(np.min(track[7:12]))
        left_mid      = float(np.mean(track[3:7]))
        right_mid     = float(np.mean(track[12:16]))
        mid_asymmetry = abs(left_mid - right_mid) / (max(left_mid, right_mid) + 1.0)

        # Base target from look-ahead table
        base_target = TARGET_SPEED
        for dist_thresh, speed_limit in LOOKAHEAD_SPEEDS:
            if forward_dist < dist_thresh:
                base_target = speed_limit
                break

        # Corner severity override
        if mid_asymmetry > ASYM_TIGHT_THRESH:
            base_target = min(base_target, ASYM_TIGHT_SPEED)
        elif mid_asymmetry > ASYM_MED_THRESH:
            base_target = min(base_target, ASYM_MED_SPEED)

        return max(40.0, base_target)

    def _compute_steer_raw(self, angle, track_pos, track, speed):
        left_cluster  = float(np.mean(track[3:7]))
        right_cluster = float(np.mean(track[12:16]))
        denom         = left_cluster + right_cluster + 1.0
        mid_asymmetry = abs(left_cluster - right_cluster) / denom

        # ── Racing line: pick target position ──────────────────────────
        if mid_asymmetry > APEX_ASYM_THRESH:
            # Turning left  → inside is left  → target negative trackPos
            # Turning right → inside is right → target positive trackPos
            if right_cluster > left_cluster:   # corner opens to the right → turning left
                target_pos = -APEX_OFFSET
            else:                              # corner opens to the left  → turning right
                target_pos =  APEX_OFFSET
        else:
            target_pos = 0.0   # straight → stay centre

        # ── Layer 1: align with road axis ──────────────────────────────
        steer = angle * STEER_ANGLE_GAIN / np.pi

        # ── Layer 2: pull toward target (apex or centre) ───────────────
        steer -= (track_pos - target_pos) * STEER_CENTER_GAIN

        # ── Layer 3: lookahead only in proper corners ──────────────────
        if mid_asymmetry > 0.15:
            lookahead_steer = (right_cluster - left_cluster) / denom
            steer += lookahead_steer * STEER_LOOKAHEAD

        # ── Speed-adaptive clamp ───────────────────────────────────────
        max_steer = float(np.clip(1.2 - speed / 350.0, 0.35, 1.0))
        steer = float(np.clip(steer, -max_steer, max_steer))

        # ── Layer 4: boundary recovery (hard override) ─────────────────
        if abs(track_pos) > TRACK_BOUNDARY:
            correction = -np.sign(track_pos) * 1.0
            steer = 0.7 * correction + 0.3 * steer
            steer = float(np.clip(steer, -1.0, 1.0))

        # ── Low-pass filter ────────────────────────────────────────────
        steer = 0.4 * steer + 0.6 * self._last_steer
        self._last_steer = steer

        return steer

    def _compute_steer_from_obs(self, obs, track_arr: np.ndarray) -> float:
        """Steering for gym_torcs namedtuple obs."""
        left_cluster  = float(np.mean(track_arr[3:7]))
        right_cluster = float(np.mean(track_arr[12:16]))
        centre_offset = (right_cluster - left_cluster) / (right_cluster + left_cluster + 1.0)
        steer = centre_offset * STEER_CENTER_GAIN
        # Boundary guard
        track_pos = float(obs.trackPos) if hasattr(obs, 'trackPos') else 0.0
        if abs(track_pos) > TRACK_BOUNDARY:
            steer = -np.sign(track_pos) * 1.0
        return float(np.clip(steer, -1.0, 1.0))

    def _auto_gear(self, speed: float, current_gear: int) -> int:
        g = max(1, min(6, current_gear))
        if g < 6 and speed > GEAR_UP[g]:
            g += 1
        elif g > 1 and speed < GEAR_DOWN[g]:
            g -= 1
        return g
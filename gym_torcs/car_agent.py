"""
car_agent.py  -  Alpha driver (v0.1)
Goal: drive forward, stay on track, don't spin out.
No ML yet — pure rule-based control.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
TARGET_SPEED      = 100     # km/h  - cruise target
MAX_ACCEL         = 0.9     # 0-1   - hard cap so we don't floor it blindly
STEER_ANGLE_GAIN  = 15.0     # how strongly to correct for angle misalignment
STEER_CENTER_GAIN = 0.2     # how strongly to pull back to track centre
BRAKE_THRESHOLD   = 30      # km/h  over target before we touch the brakes

# Gear-up / gear-down speed thresholds  (km/h)
GEAR_UP   = [0,  50,  80, 110, 140, 170]   # index = current gear (1-based → index 1..5)
GEAR_DOWN = [0,   0,  40,  65,  90, 120]


class AlphaDriver:
    """
    Minimalist alpha-version driver.
    Interface mirrors gym_torcs / snakeoil expectations:
        action = driver.act(obs)          - obs is the namedtuple from gym_torcs
        action = driver.act_raw(S_d)      - S_d is the raw snakeoil dict
    """

    def __init__(self):
        self.gear = 1

    # ------------------------------------------------------------------
    # Primary interface used by gym_torcs  (observation is a namedtuple)
    # ------------------------------------------------------------------
    def act(self, obs, reward=0.0, done=False, vision=False):
        """
        Returns a 1-D numpy array: [steer]  when throttle=False
        or [steer, accel]                    when throttle=True.
        gym_torcs handles acceleration itself when throttle=False,
        so we just return the steering signal here.
        """
        steer = self._compute_steer(
            angle=float(obs.speedX) * 0,           # placeholder - we use obs directly
            track_pos=None,
            obs=obs,
        )
        # Return [steer] - gym_torcs manages throttle in auto mode
        return np.array([steer], dtype=np.float32)

    # ------------------------------------------------------------------
    # Raw interface used when talking directly to snakeoil (run.py loop)
    # ------------------------------------------------------------------
    def act_raw(self, S, R):
        """
        Reads from snakeoil ServerState dict S.d, writes into DriverAction dict R.d.
        Call this instead of act() when you drive the snakeoil loop yourself.
        """
        s = S.d
        r = R.d

        # --- Steering ---------------------------------------------------
        # Align the car with the road axis (angle) + pull to centre (trackPos)
        steer = s['angle'] * STEER_ANGLE_GAIN / np.pi
        steer -= s['trackPos'] * STEER_CENTER_GAIN
        steer = float(np.clip(steer, -1.0, 1.0))
        r['steer'] = steer

        # --- Throttle / Brake -------------------------------------------
        speed = s['speedX']   # km/h forward speed

        if speed < TARGET_SPEED - abs(steer) * 30:
            # Accelerate - taper off as we approach target
            gap = TARGET_SPEED - speed
            r['accel'] = float(np.clip(0.05 + gap / TARGET_SPEED, 0.0, MAX_ACCEL))
            r['brake'] = 0.0
        elif speed > TARGET_SPEED + BRAKE_THRESHOLD:
            r['accel'] = 0.0
            r['brake'] = float(np.clip((speed - TARGET_SPEED) / 50.0, 0.0, 0.5))
        else:
            r['accel'] = 0.05   # coast
            r['brake'] = 0.0

        # Low-speed kick to get moving
        if speed < 5:
            r['accel'] = min(1.0, r['accel'] + 1.0 / (speed + 0.5))

        # Traction control - back off if rear wheels spin faster than fronts
        front = s['wheelSpinVel'][0] + s['wheelSpinVel'][1]
        rear  = s['wheelSpinVel'][2] + s['wheelSpinVel'][3]
        if rear - front > 5:
            r['accel'] = max(0.0, r['accel'] - 0.2)

        # --- Automatic gearbox -----------------------------------------
        r['gear'] = self._auto_gear(speed, r['gear'] if r['gear'] > 0 else 1)

        # --- Always release clutch / don't use meta --------------------
        r['clutch'] = 0.0
        r['meta']   = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_steer(self, angle, track_pos, obs):
        """Steering from namedtuple obs (used by act())."""
        raw_angle    = float(obs.speedX)   # speedX is normalised; real angle not exposed here
        # obs.track gives normalised distances; centre sensor is index 9 (straight ahead)
        track_arr = np.array(obs.track)
        track_pos_val = float(track_arr[9]) - 0.5  # crude centre offset proxy

        steer = float(obs.speedX) * 0   # start at zero — gym_torcs drives throttle
        # The observation doesn't expose angle directly in namedtuple form,
        # so we use the centre track sensor heuristic only.
        # Full angle control is handled in act_raw() for the snakeoil loop.
        steer = -track_pos_val * STEER_CENTER_GAIN
        return float(np.clip(steer, -1.0, 1.0))

    def _auto_gear(self, speed, current_gear):
        g = int(current_gear)
        g = max(1, min(6, g))
        if g < 6 and speed > GEAR_UP[g]:
            g += 1
        elif g > 1 and speed < GEAR_DOWN[g]:
            g -= 1
        return g
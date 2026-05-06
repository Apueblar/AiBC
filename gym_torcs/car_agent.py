"""
car_agent.py — AlphaDriver for TORCS / snakeoil3 / gym_torcs
=============================================================
Based directly on the proven drive_example() formula from snakeoil3_gym.py
and jmcncarai.py — the reference implementations that are KNOWN to work.

Core steering formula (from snakeoil3_gym.py drive_example):
    steer = angle * K_angle - trackPos * K_pos

This is the gold standard for TORCS controllers.

Speed is managed by reducing target speed proportional to steering demand,
exactly as the reference does: target_speed - abs(steer) * factor

All tunable constants are exposed in DEFAULT_PARAMS for the optimizer.
"""

import json
import os
import math
import numpy as np

PI = math.pi

# ---------------------------------------------------------------------------
# Load best params
# ---------------------------------------------------------------------------
_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')


def _load_saved_params() -> dict:
    if os.path.exists(_PARAMS_FILE):
        with open(_PARAMS_FILE) as f:
            p = json.load(f)
        lap = p.get('best_lap_time', '?')
        try:
            lap_str = f'{float(lap):.2f}s'
        except (TypeError, ValueError):
            lap_str = str(lap)
        print(f'[AlphaDriver] Loaded best_params.json  (best_lap={lap_str})')
        return p
    print('[AlphaDriver] No best_params.json — using hardcoded defaults.')
    return {}


_SAVED = _load_saved_params()
_s = lambda key, default: _SAVED.get(key, default)   # noqa: E731


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Steering — core PD formula: steer = angle * K_angle - trackPos * K_pos
K_ANGLE             = _s('K_ANGLE',          15.0 / PI)  # ~4.77 (from reference)
K_POS               = _s('K_POS',            0.10)

# Speed management
TARGET_SPEED        = _s('TARGET_SPEED',      280.0)
# Target speed is reduced by abs(steer) * STEER_SPEED_FACTOR
STEER_SPEED_FACTOR  = _s('STEER_SPEED_FACTOR', 100.0)  # steer of 0.5 → reduce speed by 50

# Gear thresholds by speed (km/h)
GEAR_UP             = _s('GEAR_UP',   [0,  50,  80, 110, 140, 170])
GEAR_DOWN           = _s('GEAR_DOWN', [0,   0,  35,  65,  95, 130])


# ---------------------------------------------------------------------------
# Convenience: default params dict
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    'K_ANGLE':            K_ANGLE,
    'K_POS':              K_POS,
    'TARGET_SPEED':       TARGET_SPEED,
    'STEER_SPEED_FACTOR': STEER_SPEED_FACTOR,
    'GEAR_UP':            list(GEAR_UP),
    'GEAR_DOWN':          list(GEAR_DOWN),
}


# ===========================================================================
class AlphaDriver:
# ===========================================================================
    """
    Race driver for TORCS via snakeoil3 (act_raw) or gym_torcs (act).

    Uses the exact steering formula from the reference snakeoil3_gym.py
    drive_example that is known to successfully lap every TORCS track.
    """

    def __init__(self, params: dict | None = None):
        self.p = dict(DEFAULT_PARAMS) if params is None else params
        self._reset_state()

    def reset(self):
        """Call at the start of each episode to clear per-lap state."""
        self._reset_state()

    def _reset_state(self):
        self.gear = 1

    # --------------------------------------------------------- gym_torcs API
    def act(self, obs, reward=0.0, done=False, vision=False):
        """gym_torcs interface."""
        steer = self._steer(float(obs.angle), float(obs.trackPos))
        return np.array([steer], dtype=np.float32)

    # -------------------------------------------------------- snakeoil3 API
    def act_raw(self, S, R):
        """snakeoil3 interface."""
        p  = self.p
        s  = S.d
        r  = R.d

        speed     = float(s['speedX'])
        angle     = float(s['angle'])
        track_pos = float(s['trackPos'])

        # -- Steering (reference formula) ----------------------------------
        steer = self._steer(angle, track_pos)
        r['steer'] = steer

        # -- Corner anticipation (track sensors) --------------------------
        # Use track sensors to detect corners BEFORE the car enters them.
        # Asymmetry between left/right beams = upcoming corner sharpness.
        # Beams: 0-3 are left (-45° to -7°), 15-18 are right (7° to 45°)
        track = np.array(s['track'], dtype=np.float32)
        safe = np.where(track < 0, 200.0, track)

        left_avg  = float(np.mean(safe[0:4]))    # -45° to -7°
        right_avg = float(np.mean(safe[15:19]))  # +7° to +45°
        corner_asym = abs(left_avg - right_avg)

        # Forward clearance: 75th percentile of beams 5-13 (±2.5° to ±4°)
        # 75th percentile ignores individual glitchy beams (like ramp hits)
        fwd_p75 = float(np.percentile(safe[5:14], 75))

        # Speed caps:
        # 1. Forward clearance cap
        fwd_cap = float(np.interp(fwd_p75,
            p.get('CORNER_DISTS',  [5,  15,  30,  50,  80, 120, 200]),
            p.get('CORNER_SPEEDS', [40,  55,  80, 110, 150, 200, 280])
        ))

        # 2. Asymmetry cap — more asymmetric = sharper corner = slower
        asym_cap = float(np.interp(corner_asym,
            p.get('ASYM_BREAKS',  [0,  15,  35,  60,  90, 130]),
            p.get('ASYM_SPEEDS',  [280, 220, 160, 110,  70,  50])
        ))

        corner_target = min(fwd_cap, asym_cap, float(p['TARGET_SPEED']))

        # -- Speed target: combined corner + steer-based reduction ---------
        steer_target = float(p['TARGET_SPEED']) - abs(steer) * float(p['STEER_SPEED_FACTOR'])
        target = min(corner_target, max(40.0, steer_target))

        # -- Throttle / Brake ---------------------------------------------
        if speed < target:
            # Much more aggressive acceleration: full throttle if we're 5km/h or more below target
            if target - speed > 5.0:
                r['accel'] = 1.0
            else:
                r['accel'] = min(1.0, (target - speed) / 5.0 + 0.5)
            r['brake'] = 0.0
        else:
            r['accel'] = 0.0
            r['brake'] = min(1.0, (speed - target) / 50.0)

        # Low-speed launch
        if speed < 10:
            r['accel'] = 1.0
            r['brake'] = 0.0

        # -- Traction control (from reference) -----------------------------
        if ((s['wheelSpinVel'][2] + s['wheelSpinVel'][3]) -
                (s['wheelSpinVel'][0] + s['wheelSpinVel'][1]) > 5):
            r['accel'] -= 0.2

        r['accel'] = max(0.0, min(1.0, r['accel']))

        # -- Gear (speed-based, from reference jmcncarai.py) --------------
        cur_g = max(1, min(6, int(s.get('gear', self.gear)) or 1))
        gu = p['GEAR_UP']
        gd = p['GEAR_DOWN']
        if cur_g < 6 and speed > gu[cur_g]:
            cur_g += 1
        elif cur_g > 1 and speed < gd[cur_g]:
            cur_g -= 1
        r['gear']   = cur_g
        self.gear   = cur_g
        r['clutch'] = 0.0
        r['meta']   = 0

    # ---------------------------------------- core steering
    def _steer(self, angle: float, track_pos: float) -> float:
        """
        Reference formula from snakeoil3_gym.py drive_example:
            steer = angle * (15/PI) - trackPos * 0.10

        K_ANGLE = 15/PI ≈ 4.77  gives full lock at ~12° off-heading
        K_POS   = 0.10           gentle centering
        """
        p = self.p
        steer = float(p['K_ANGLE']) * angle - float(p['K_POS']) * track_pos
        return float(np.clip(steer, -1.0, 1.0))
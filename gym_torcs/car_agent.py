"""
car_agent.py — AlphaDriver for TORCS / snakeoil3 / gym_torcs
=============================================================
* Loads tuned params from best_params.json when present; falls back to
  hardcoded defaults otherwise.
* AlphaDriver(params=None)  — pass a dict to override every default at once.
  Used by improver.py so there is no duplicated driver code.

New vs. original
----------------
- Wall-avoidance layer: soft bias + hard emergency override in steering
- Wall-proximity speed cap in adaptive target
- Steer smoothing (STEER_SMOOTH, default 0 = off)
- All constants exposed as dict → trivially overridable by the optimizer
"""

import json
import os
import numpy as np

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
# Module-level defaults  (overridden by best_params.json when present)
# ---------------------------------------------------------------------------
TARGET_SPEED        = _s('TARGET_SPEED',        360.0)
MAX_ACCEL           = _s('MAX_ACCEL',           1.0)

# Steering
STEER_ANGLE_GAIN    = _s('STEER_ANGLE_GAIN',    0.55)
STEER_CENTER_GAIN   = _s('STEER_CENTER_GAIN',   0.40)
STEER_LOOKAHEAD     = _s('STEER_LOOKAHEAD',     0.25)
STEER_SPEED_DENOM   = _s('STEER_SPEED_DENOM',   500.0)
MIN_STEER_SCALE     = _s('MIN_STEER_SCALE',     0.25)
STEER_SMOOTH        = _s('STEER_SMOOTH',        0.10)   # blend with prev steer

# Lookahead speed curve  (track[7:12] forward distance → speed cap)
LOOKAHEAD_SPEEDS    = _s('LOOKAHEAD_SPEEDS',    [45,  70, 115, 155, 200, 270, 360])
LOOKAHEAD_DISTS     = _s('LOOKAHEAD_DISTS',     [10,  22,  40,  65, 100, 150, 200])

# Corner speed curve  (left/right asymmetry → speed cap)
CORNER_ASYMS        = _s('CORNER_ASYMS',        [  0,   8,  20,  35,  52,  70])
CORNER_SPEEDS       = _s('CORNER_SPEEDS',       [360, 270, 155,  95,  62,  48])

# Speed-target slew
TARGET_SLEW_DOWN    = _s('TARGET_SLEW_DOWN',    4.0)
TARGET_SLEW_UP      = _s('TARGET_SLEW_UP',      8.0)

# Apex / racing line
APEX_OFFSET         = _s('APEX_OFFSET',         0.55)
APEX_ASYM_THRESH    = _s('APEX_ASYM_THRESH',    6.0)
TARGET_POS_SLEW     = _s('TARGET_POS_SLEW',     0.10)

# Braking
BRAKE_GAIN          = _s('BRAKE_GAIN',          0.08)
MAX_BRAKE           = _s('MAX_BRAKE',           0.90)
BRAKE_TRIGGER       = _s('BRAKE_TRIGGER',       2.0)

# Forward-clearance / traction control
PTP_DIST_THRESHOLD  = _s('PTP_DIST_THRESHOLD',  60.0)
TC_THRESHOLD        = _s('TC_THRESHOLD',        50.0)
TC_REDUCTION        = _s('TC_REDUCTION',        0.20)

# Wall avoidance  — these are SAFETY params, not tuned by the optimizer
WALL_DANGER_DIST    = _s('WALL_DANGER_DIST',    4.0)    # m  — hard steer override
WALL_WARN_DIST      = _s('WALL_WARN_DIST',      10.0)   # m  — soft bias + throttle cut
WALL_DANGER_GAIN    = _s('WALL_DANGER_GAIN',    0.65)   # override steer magnitude
WALL_WARN_GAIN      = _s('WALL_WARN_GAIN',      0.20)   # bias gain (per unit proximity)

# Gear thresholds  (index = current gear; compare against speedX km/h)
GEAR_UP             = _s('GEAR_UP',   [0,  55, 100, 145, 190, 240])
GEAR_DOWN           = _s('GEAR_DOWN', [0,   0,  40,  80, 120, 165])


# ---------------------------------------------------------------------------
# Convenience: default params dict (used by AlphaDriver() and by the improver
# as a warm-start / fall-back value)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    'TARGET_SPEED':       TARGET_SPEED,
    'MAX_ACCEL':          MAX_ACCEL,
    'STEER_ANGLE_GAIN':   STEER_ANGLE_GAIN,
    'STEER_CENTER_GAIN':  STEER_CENTER_GAIN,
    'STEER_LOOKAHEAD':    STEER_LOOKAHEAD,
    'STEER_SPEED_DENOM':  STEER_SPEED_DENOM,
    'MIN_STEER_SCALE':    MIN_STEER_SCALE,
    'STEER_SMOOTH':       STEER_SMOOTH,
    'LOOKAHEAD_SPEEDS':   list(LOOKAHEAD_SPEEDS),
    'LOOKAHEAD_DISTS':    list(LOOKAHEAD_DISTS),
    'CORNER_ASYMS':       list(CORNER_ASYMS),
    'CORNER_SPEEDS':      list(CORNER_SPEEDS),
    'TARGET_SLEW_DOWN':   TARGET_SLEW_DOWN,
    'TARGET_SLEW_UP':     TARGET_SLEW_UP,
    'APEX_OFFSET':        APEX_OFFSET,
    'APEX_ASYM_THRESH':   APEX_ASYM_THRESH,
    'TARGET_POS_SLEW':    TARGET_POS_SLEW,
    'BRAKE_GAIN':         BRAKE_GAIN,
    'MAX_BRAKE':          MAX_BRAKE,
    'BRAKE_TRIGGER':      BRAKE_TRIGGER,
    'PTP_DIST_THRESHOLD': PTP_DIST_THRESHOLD,
    'TC_THRESHOLD':       TC_THRESHOLD,
    'TC_REDUCTION':       TC_REDUCTION,
    'WALL_DANGER_DIST':   WALL_DANGER_DIST,
    'WALL_WARN_DIST':     WALL_WARN_DIST,
    'WALL_DANGER_GAIN':   WALL_DANGER_GAIN,
    'WALL_WARN_GAIN':     WALL_WARN_GAIN,
    'GEAR_UP':            list(GEAR_UP),
    'GEAR_DOWN':          list(GEAR_DOWN),
}


# ===========================================================================
class AlphaDriver:
# ===========================================================================
    """
    Race driver for TORCS via snakeoil3 (act_raw) or gym_torcs (act).

    Parameters
    ----------
    params : dict or None
        When None  → uses DEFAULT_PARAMS (derived from best_params.json or
                      hardcoded values above).
        When dict  → uses that dict verbatim; every key in DEFAULT_PARAMS must
                      be present.  Used by improver.py to inject trial params.
    """

    def __init__(self, params: dict | None = None):
        self.p = dict(DEFAULT_PARAMS) if params is None else params
        self._reset_state()

    # ------------------------------------------------------------------ reset
    def reset(self):
        """Call at the start of each episode to clear per-lap state."""
        self._reset_state()

    def _reset_state(self):
        self.gear           = 1
        self._target_pos    = 0.0
        self._smooth_target = float(self.p['TARGET_SPEED'])
        self._last_steer    = 0.0

    # --------------------------------------------------------- gym_torcs API
    def act(self, obs, reward=0.0, done=False, vision=False):
        """gym_torcs interface: obs is a CarState / Observation object."""
        track_arr = np.array(obs.track, dtype=np.float32)
        steer = self._compute_steer(
            angle     = float(obs.angle),
            track_pos = float(obs.trackPos),
            track     = track_arr,
            speed     = float(obs.speedX),
        )
        return np.array([steer], dtype=np.float32)

    # -------------------------------------------------------- snakeoil3 API
    def act_raw(self, S, R):
        """
        snakeoil3 interface.
        Reads from S.d (sensor dict), writes into R.d (response dict).
        """
        p = self.p
        s = S.d
        r = R.d

        track     = np.array(s['track'],  dtype=np.float32)
        speed     = float(s['speedX'])
        angle     = float(s['angle'])
        track_pos = float(s['trackPos'])

        # ── Adaptive speed target (slewed for smoothness) ─────────────────
        raw_target = self._adaptive_target(track)
        delta      = raw_target - self._smooth_target
        if delta < 0:
            self._smooth_target += max(delta, -float(p['TARGET_SLEW_DOWN']))
        else:
            self._smooth_target += min(delta,  float(p['TARGET_SLEW_UP']))
        target = self._smooth_target

        # ── Steering ─────────────────────────────────────────────────────
        r['steer'] = self._compute_steer(angle, track_pos, track, speed)

        # ── Throttle / brake ──────────────────────────────────────────────
        speed_err = speed - target
        fwd_beams = np.where(track[7:12] < 0, 200.0, track[7:12])
        forward_clear = float(np.min(fwd_beams)) > float(p['PTP_DIST_THRESHOLD'])

        if speed_err < -float(p['BRAKE_TRIGGER']):
            gap   = target - speed
            if forward_clear:
                accel = float(p['MAX_ACCEL'])
            else:
                accel = float(np.clip(
                    0.15 + gap / (target + 1.0), 0.0, float(p['MAX_ACCEL'])
                ))
                if speed_err > -25:
                    accel *= 0.65
            r['accel'] = accel
            r['brake'] = 0.0
        elif speed_err > float(p['BRAKE_TRIGGER']):
            r['accel'] = 0.0
            r['brake'] = float(np.clip(
                speed_err * float(p['BRAKE_GAIN']), 0.0, float(p['MAX_BRAKE'])
            ))
        else:
            r['accel'] = 0.18
            r['brake'] = 0.0

        # Low-speed launch boost
        if speed < 5:
            r['accel'] = min(1.0, r['accel'] + 1.0 / (speed + 0.5))
            r['brake'] = 0.0

        # ── Traction control ──────────────────────────────────────────────
        front = s['wheelSpinVel'][0] + s['wheelSpinVel'][1]
        rear  = s['wheelSpinVel'][2] + s['wheelSpinVel'][3]
        if rear - front > float(p['TC_THRESHOLD']):
            r['accel'] = max(0.0, r['accel'] - float(p['TC_REDUCTION']))

        # ── Wall proximity → throttle reduction ───────────────────────────
        safe  = np.where(track < 0, 200.0, track)
        lwall = float(np.min(safe[0:3]))    # leftmost 3 beams  (-90° … -70°)
        rwall = float(np.min(safe[16:19]))  # rightmost 3 beams (+70° … +90°)
        wall_min = min(lwall, rwall)
        wd = float(p['WALL_WARN_DIST'])
        if wall_min < wd and speed > 40:
            proximity = 1.0 - wall_min / wd          # 0 at edge, 1 at wall
            throttle_cut = max(0.0, proximity - 0.3) # only cut when >30% proximity
            r['accel'] = max(0.0, r['accel'] * (1.0 - throttle_cut))
            if wall_min < float(p['WALL_DANGER_DIST']):
                r['brake'] = max(r['brake'], proximity * 0.35)

        # ── Gear ─────────────────────────────────────────────────────────
        cur_g = max(1, min(6, int(s.get('gear', self.gear)) or 1))
        gu, gd = p['GEAR_UP'], p['GEAR_DOWN']
        if cur_g < 6 and speed > gu[cur_g]:   cur_g += 1
        elif cur_g > 1 and speed < gd[cur_g]: cur_g -= 1
        r['gear']   = cur_g
        self.gear   = cur_g
        r['clutch'] = 0.0
        r['meta']   = 0

    # ---------------------------------------------- adaptive speed target
    def _adaptive_target(self, track: np.ndarray) -> float:
        p = self.p

        # Forward clearance cap
        fwd     = np.where(track[7:12] < 0, 200.0, track[7:12])
        fwd_cap = float(np.interp(
            float(np.min(fwd)), p['LOOKAHEAD_DISTS'], p['LOOKAHEAD_SPEEDS']
        ))

        # Corner asymmetry cap
        side    = np.where(track < 0, 200.0, track)
        asym    = abs(float(np.mean(side[3:7])) - float(np.mean(side[12:16])))
        crn_cap = float(np.interp(asym, p['CORNER_ASYMS'], p['CORNER_SPEEDS']))

        # Wall proximity cap
        lwall   = float(np.min(side[0:3]))
        rwall   = float(np.min(side[16:19]))
        wd      = float(p['WALL_WARN_DIST'])
        wall    = min(lwall, rwall)
        wall_cap = (float(p['TARGET_SPEED']) * max(0.35, wall / wd)
                    if wall < wd else float(p['TARGET_SPEED']))

        return float(min(fwd_cap, crn_cap, wall_cap, p['TARGET_SPEED']))

    # -------------------------------------------------------- steering law
    def _compute_steer(
        self,
        angle: float,
        track_pos: float,
        track: np.ndarray,
        speed: float,
    ) -> float:
        p    = self.p
        safe = np.where(track < 0, 200.0, track)

        # ── Racing-line apex target ───────────────────────────────────────
        signed_asym = float(np.mean(safe[3:7])) - float(np.mean(safe[12:16]))
        if abs(signed_asym) > float(p['APEX_ASYM_THRESH']):
            scale   = float(np.clip(abs(signed_asym) / 60.0, 0.0, 1.0))
            apt_raw = (signed_asym / 100.0) * float(p['APEX_OFFSET']) * (1.0 + scale)
            apex_target = float(np.clip(
                apt_raw, -float(p['APEX_OFFSET']), float(p['APEX_OFFSET'])
            ))
        else:
            apex_target = 0.0
        self._target_pos += float(p['TARGET_POS_SLEW']) * (apex_target - self._target_pos)

        # ── Lookahead feed-forward ────────────────────────────────────────
        fc  = max(float(track[9]), 0.0)   # beam 9 = straight ahead
        fl  = max(float(track[7]), 0.0)   # beam 7 = slight left
        lah = float(p['STEER_LOOKAHEAD']) * (fl - fc) / (fc + 1.0)

        # ── Base PD steer ─────────────────────────────────────────────────
        steer = (float(p['STEER_ANGLE_GAIN'])  * angle
               - float(p['STEER_CENTER_GAIN']) * (track_pos - self._target_pos)
               + lah)

        # ── Speed-dependent scaling ───────────────────────────────────────
        speed_scale = max(
            float(p['MIN_STEER_SCALE']),
            1.0 - speed / float(p['STEER_SPEED_DENOM'])
        )
        steer *= speed_scale

        # ── Wall-avoidance soft bias ──────────────────────────────────────
        # track[0:3]   = left-side beams  (-90°, -80°, -70°)
        # track[16:19] = right-side beams (+70°, +80°, +90°)
        lwall = float(np.min(safe[0:3]))
        rwall = float(np.min(safe[16:19]))
        wd    = float(p['WALL_WARN_DIST'])
        dd    = float(p['WALL_DANGER_DIST'])
        wg    = float(p['WALL_WARN_GAIN'])
        dg    = float(p['WALL_DANGER_GAIN'])

        # Soft bias: push away from whichever wall is closer
        if lwall < wd:                           # steer right (negative)
            steer -= wg * (1.0 - lwall / wd)
        if rwall < wd:                           # steer left  (positive)
            steer += wg * (1.0 - rwall / wd)

        # Hard emergency override (very close to wall — takes precedence)
        if lwall < dd:
            emergency = -dg * (1.0 - lwall / dd)
            steer = min(steer, emergency)        # pick the more-rightward value
        if rwall < dd:
            emergency = dg * (1.0 - rwall / dd)
            steer = max(steer, emergency)        # pick the more-leftward value

        steer = float(np.clip(steer, -1.0, 1.0))

        # ── Steer smoothing ───────────────────────────────────────────────
        sm    = float(np.clip(p.get('STEER_SMOOTH', 0.0), 0.0, 0.9))
        steer = (1.0 - sm) * steer + sm * self._last_steer
        steer = float(np.clip(steer, -1.0, 1.0))

        self._last_steer = steer
        return steer
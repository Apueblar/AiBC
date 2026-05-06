"""
sensor_processor.py  —  TORCS Sensor Processing Module
IBM AI Racing League – Country Challenge 2026

Responsibilities
----------------
1. Parse and validate the raw sensor dict from snakeoil3 (client.S.d)
2. Normalise every sensor to a consistent numeric range  [0, 1] or [-1, 1]
3. Compute derived / engineered features useful for AI decision-making
4. Package everything into a structured SensorObservation dataclass

Usage (snakeoil loop — run.py style)
-------------------------------------
    from sensor_processor import SensorProcessor, SensorLogger

    processor = SensorProcessor()
    logger    = SensorLogger(output_dir='../results/sensors')

    # --- start of episode ---
    processor.reset()
    logger.open(episode=1)              # creates  results/sensors/ep001_<timestamp>.csv

    client.get_servers_input()
    obs = processor.process(client.S.d)
    logger.write(obs, episode=1, step=0)  # one row per step

    print(obs.summary())                # human-readable one-liner
    print(obs.speed_x)                  # raw forward speed   (km/h)
    print(obs.speed_x_norm)             # normalised  [0, 1]
    print(obs.track_sensors_norm)       # 19-element numpy array
    print(obs.is_near_wall)             # True / False
    vec = obs.to_vector()               # flat float32 array for a neural net

    # --- end of episode ---
    logger.close()                      # flushes and closes the CSV

Usage (gym_torcs loop)
----------------------
    raw_obs = client.S.d               # same underlying dict
    obs = processor.process(raw_obs)
"""

import csv
import datetime
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

# Angles (degrees) of each of the 19 track range-finder beams
TRACK_SENSOR_ANGLES_DEG: list = [
    -45, -19, -12, -7, -4, -2.5, -1.7, -1, -0.5,
      0,
     0.5,   1,  1.7, 2.5, 4,    7,   12, 19, 45,
]
N_TRACK_SENSORS      = 19
N_OPPONENT_SENSORS   = 36
N_FOCUS_SENSORS      = 5
N_WHEEL_SENSORS      = 4   # order: FL, FR, RL, RR
TRACK_CENTER_IDX     = 9   # index of the dead-ahead sensor

# Normalisation denominators
MAX_SPEED_KMH    = 300.0   # approximate top speed in TORCS  (km/h)
MAX_TRACK_DIST_M = 200.0   # maximum track sensor range       (m)
MAX_OPP_DIST_M   = 200.0   # maximum opponent sensor range    (m)
MAX_RPM          = 10_000.0
MAX_FUEL         = 100.0
MAX_DAMAGE       = 10_000.0
MAX_WHEEL_VEL    = 100.0   # rad/s — empirical upper bound

# Derived-feature thresholds (all tunable via SensorProcessor.__init__)
DEFAULT_NEAR_WALL_M      = 5.0    # track sensor below this  → near wall (m)
DEFAULT_STUCK_SPEED_KMH  = 3.0    # forward speed below this → potentially stuck
DEFAULT_SLIP_THRESHOLD   = 5.0    # rear−front wheel diff    → spinning (rad/s)
DEFAULT_ALIGN_RAD        = 0.1    # abs(angle) below this    → car is aligned (rad)


# ---------------------------------------------------------------------------
# Observation dataclass
# ---------------------------------------------------------------------------

@dataclass
class SensorObservation:
    """
    One fully processed TORCS timestep.

    Naming conventions
    ------------------
    *           - raw sensor value with physical units
    *_norm      - clipped and normalised to [0, 1] or [-1, 1]
    is_*        - boolean derived flag
    (no suffix) - computed metric (same units as raw)
    """

    # ── Raw scalars ────────────────────────────────────────────────────────
    angle_rad:       float = 0.0   # rad  — car angle vs track axis
    track_pos:       float = 0.0   # [-1 left .. +1 right]
    speed_x:         float = 0.0   # km/h — forward
    speed_y:         float = 0.0   # km/h — lateral (positive = rightward)
    speed_z:         float = 0.0   # km/h — vertical
    rpm:             float = 0.0
    gear:            int   = 1     # -1=R  0=N  1-6=drive
    fuel:            float = 100.0
    damage:          float = 0.0
    dist_raced:      float = 0.0   # m — total distance driven this session
    dist_from_start: float = 0.0   # m — position along current lap
    cur_lap_time:    float = 0.0   # s
    last_lap_time:   float = 0.0   # s — 0 until the first lap is done
    race_pos:        int   = 1
    z:               float = 0.3   # m — height above road surface

    # ── Raw arrays ─────────────────────────────────────────────────────────
    track_sensors:    np.ndarray = field(default_factory=lambda: np.full(N_TRACK_SENSORS,    200.0))
    opponent_sensors: np.ndarray = field(default_factory=lambda: np.full(N_OPPONENT_SENSORS, 200.0))
    wheel_spin_vel:   np.ndarray = field(default_factory=lambda: np.zeros(N_WHEEL_SENSORS))
    focus_sensors:    np.ndarray = field(default_factory=lambda: np.zeros(N_FOCUS_SENSORS))

    # ── Normalised scalars ─────────────────────────────────────────────────
    angle_norm:     float = 0.0   # angle / π            → [-1, 1]
    track_pos_norm: float = 0.0   # same as track_pos    → [-1, 1]
    speed_x_norm:   float = 0.0   # speed_x / MAX        → [ 0, 1]
    speed_y_norm:   float = 0.0   # speed_y / MAX        → [-1, 1]
    speed_z_norm:   float = 0.0   # speed_z / MAX        → [-1, 1]
    rpm_norm:       float = 0.0   # rpm / MAX            → [ 0, 1]
    fuel_norm:      float = 0.0   # fuel / MAX           → [ 0, 1]
    damage_norm:    float = 0.0   # damage / MAX         → [ 0, 1]

    # ── Normalised arrays ──────────────────────────────────────────────────
    track_sensors_norm:    np.ndarray = field(default_factory=lambda: np.zeros(N_TRACK_SENSORS))
    opponent_sensors_norm: np.ndarray = field(default_factory=lambda: np.zeros(N_OPPONENT_SENSORS))
    wheel_spin_vel_norm:   np.ndarray = field(default_factory=lambda: np.zeros(N_WHEEL_SENSORS))
    focus_sensors_norm:    np.ndarray = field(default_factory=lambda: np.zeros(N_FOCUS_SENSORS))

    # ── Derived / engineered features ──────────────────────────────────────
    effective_speed:          float = 0.0    # speed_x * cos(angle) — track-axis progress (km/h)
    speed_magnitude:          float = 0.0    # sqrt(Vx² + Vy²)                           (km/h)
    track_ahead:              float = 200.0  # centre beam distance — space ahead          (m)
    track_left_clearance:     float = 200.0  # min left-side sensor                        (m)
    track_right_clearance:    float = 200.0  # min right-side sensor                       (m)
    closest_opponent:         float = 200.0  # min of all 36 opponent distances            (m)
    slip_ratio:               float = 0.0    # rear − front wheel speed ≥0 (traction loss) (rad/s)
    is_near_wall:             bool  = False  # any track sensor < near_wall_threshold
    is_going_forward:         bool  = True   # cos(angle) > 0
    is_car_aligned:           bool  = True   # |angle| < align threshold
    is_spinning:              bool  = False  # slip_ratio > slip threshold
    recommended_gear:         int   = 1      # speed-to-gear heuristic

    # ── to_vector() dimension info ─────────────────────────────────────────
    # Without opponents: 8 scalars + 19 track + 4 wheel + 10 derived = 41
    # With    opponents: 41 + 36                                       = 77

    def to_vector(self, include_opponents: bool = False) -> np.ndarray:
        """
        Flatten the observation to a 1-D float32 numpy array.

        Suitable for feeding directly into a neural network or tabular model.

        Parameters
        ----------
        include_opponents : bool
            Whether to append the 36 normalised opponent sensors.
            Default False keeps the vector compact for simpler models.

        Returns
        -------
        np.ndarray, shape (41,) or (77,)

        Feature layout
        --------------
        [ 0]     angle_norm
        [ 1]     track_pos_norm
        [ 2]     speed_x_norm
        [ 3]     speed_y_norm
        [ 4]     speed_z_norm
        [ 5]     rpm_norm
        [ 6]     fuel_norm
        [ 7]     damage_norm
        [ 8-26]  track_sensors_norm      (19)
        [27-30]  wheel_spin_vel_norm      (4)
        [31]     effective_speed_norm
        [32]     track_ahead_norm
        [33]     track_left_clearance_norm
        [34]     track_right_clearance_norm
        [35]     slip_ratio_norm
        [36]     is_near_wall            (0 / 1)
        [37]     is_going_forward        (0 / 1)
        [38]     is_car_aligned          (0 / 1)
        [39]     is_spinning             (0 / 1)
        [40]     gear_norm               (gear / 6)
        [41-76]  opponent_sensors_norm   (36) — only if include_opponents=True
        """
        scalars = np.array([
            self.angle_norm,
            self.track_pos_norm,
            self.speed_x_norm,
            self.speed_y_norm,
            self.speed_z_norm,
            self.rpm_norm,
            self.fuel_norm,
            self.damage_norm,
            float(np.clip(self.effective_speed    / MAX_SPEED_KMH,    -1.0, 1.0)),
            float(np.clip(self.track_ahead         / MAX_TRACK_DIST_M,  0.0, 1.0)),
            float(np.clip(self.track_left_clearance / MAX_TRACK_DIST_M, 0.0, 1.0)),
            float(np.clip(self.track_right_clearance/ MAX_TRACK_DIST_M, 0.0, 1.0)),
            float(np.clip(self.slip_ratio          / DEFAULT_SLIP_THRESHOLD, 0.0, 1.0)),
            float(self.is_near_wall),
            float(self.is_going_forward),
            float(self.is_car_aligned),
            float(self.is_spinning),
            float(np.clip(self.gear / 6.0, -1.0 / 6.0, 1.0)),
        ], dtype=np.float32)

        parts = [
            scalars,
            self.track_sensors_norm.astype(np.float32),
            self.wheel_spin_vel_norm.astype(np.float32),
        ]
        if include_opponents:
            parts.append(self.opponent_sensors_norm.astype(np.float32))

        return np.concatenate(parts)

    def summary(self) -> str:
        """Compact human-readable string — useful for logging / debugging."""
        flags = ''
        if self.is_near_wall:       flags += 'WALL '
        if not self.is_going_forward: flags += 'BACKWARD '
        if self.is_spinning:        flags += 'SLIP '
        if not self.is_car_aligned: flags += 'MISALIGNED '
        return (
            f"spd={self.speed_x:5.1f}km/h  "
            f"angle={np.degrees(self.angle_rad):+6.1f}deg  "
            f"pos={self.track_pos:+.3f}  "
            f"ahead={self.track_ahead:6.1f}m  "
            f"L={self.track_left_clearance:5.1f}m  "
            f"R={self.track_right_clearance:5.1f}m  "
            f"opp={self.closest_opponent:5.1f}m  "
            f"gear={self.gear}  "
            f"rpm={self.rpm:5.0f}  "
            f"dmg={self.damage:.0f}  "
            f"{flags}"
        )


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class SensorProcessor:
    """
    Converts a raw snakeoil3 sensor dict (client.S.d) into a fully
    populated SensorObservation with normalised values and derived features.

    Parameters
    ----------
    near_wall_threshold : float
        Track sensor distance (m) below which ``is_near_wall`` is True.
    stuck_speed_threshold : float
        Forward speed (km/h) below which the car is considered stuck.
    slip_diff_threshold : float
        Rear−front wheel speed difference (rad/s) above which ``is_spinning`` is True.
    straight_angle_threshold : float
        |angle| (rad) below which ``is_car_aligned`` is True.

    Example
    -------
        proc = SensorProcessor(near_wall_threshold=3.0)
        obs  = proc.process(client.S.d)
        nn_input = obs.to_vector()
    """

    def __init__(
        self,
        near_wall_threshold:   float = DEFAULT_NEAR_WALL_M,
        stuck_speed_threshold: float = DEFAULT_STUCK_SPEED_KMH,
        slip_diff_threshold:   float = DEFAULT_SLIP_THRESHOLD,
        straight_angle_threshold: float = DEFAULT_ALIGN_RAD,
    ):
        self.near_wall_threshold      = near_wall_threshold
        self.stuck_speed_threshold    = stuck_speed_threshold
        self.slip_diff_threshold      = slip_diff_threshold
        self.straight_angle_threshold = straight_angle_threshold

        self._prev_obs: Optional[SensorObservation] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw: dict) -> SensorObservation:
        """
        Parse, normalise, and derive features from a raw snakeoil3 dict.

        Parameters
        ----------
        raw : dict
            ``client.S.d`` — the sensor dictionary received from TORCS.

        Returns
        -------
        SensorObservation
            Fully populated observation.  Previous observation is stored
            internally for future delta / temporal features.
        """
        obs = SensorObservation()
        self._parse_raw(raw, obs)
        self._normalise(obs)
        self._derive_features(obs)
        self._prev_obs = obs
        return obs

    def reset(self):
        """
        Clear internal state.  Call at the beginning of each episode so
        delta features do not bleed across episodes.
        """
        self._prev_obs = None

    @property
    def prev_obs(self) -> Optional[SensorObservation]:
        """The observation from the previous timestep, or None."""
        return self._prev_obs

    # ------------------------------------------------------------------
    # Step 1 — parse
    # ------------------------------------------------------------------

    def _parse_raw(self, raw: dict, obs: SensorObservation) -> None:
        """
        Extract every sensor from ``raw``, applying safe defaults for
        missing or malformed keys so the processor never raises on
        partial packets.
        """
        obs.angle_rad       = float(raw.get('angle',          0.0))
        obs.track_pos       = float(raw.get('trackPos',        0.0))
        obs.speed_x         = float(raw.get('speedX',          0.0))
        obs.speed_y         = float(raw.get('speedY',          0.0))
        obs.speed_z         = float(raw.get('speedZ',          0.0))
        obs.rpm             = float(raw.get('rpm',             0.0))
        obs.gear            = int(  raw.get('gear',            1))
        obs.fuel            = float(raw.get('fuel',            100.0))
        obs.damage          = float(raw.get('damage',          0.0))
        obs.dist_raced      = float(raw.get('distRaced',       0.0))
        obs.dist_from_start = float(raw.get('distFromStart',   0.0))
        obs.cur_lap_time    = float(raw.get('curLapTime',      0.0))
        obs.last_lap_time   = float(raw.get('lastLapTime',     0.0))
        obs.race_pos        = int(  raw.get('racePos',         1))
        obs.z               = float(raw.get('z',               0.3))

        obs.track_sensors    = self._to_array(raw.get('track',        [200.0] * N_TRACK_SENSORS),
                                              expected_len=N_TRACK_SENSORS,    fill=200.0)
        obs.opponent_sensors = self._to_array(raw.get('opponents',    [200.0] * N_OPPONENT_SENSORS),
                                              expected_len=N_OPPONENT_SENSORS, fill=200.0)
        obs.wheel_spin_vel   = self._to_array(raw.get('wheelSpinVel', [0.0]   * N_WHEEL_SENSORS),
                                              expected_len=N_WHEEL_SENSORS,    fill=0.0)
        obs.focus_sensors    = self._to_array(raw.get('focus',        [0.0]   * N_FOCUS_SENSORS),
                                              expected_len=N_FOCUS_SENSORS,    fill=0.0)

    # ------------------------------------------------------------------
    # Step 2 — normalise
    # ------------------------------------------------------------------

    def _normalise(self, obs: SensorObservation) -> None:
        """Scale every raw sensor to a standard range."""

        # Scalars
        obs.angle_norm     = float(np.clip(obs.angle_rad / np.pi,           -1.0,  1.0))
        obs.track_pos_norm = float(np.clip(obs.track_pos,                   -1.0,  1.0))
        obs.speed_x_norm   = float(np.clip(obs.speed_x   / MAX_SPEED_KMH,   0.0,  1.0))
        obs.speed_y_norm   = float(np.clip(obs.speed_y   / MAX_SPEED_KMH,  -1.0,  1.0))
        obs.speed_z_norm   = float(np.clip(obs.speed_z   / MAX_SPEED_KMH,  -1.0,  1.0))
        obs.rpm_norm       = float(np.clip(obs.rpm        / MAX_RPM,         0.0,  1.0))
        obs.fuel_norm      = float(np.clip(obs.fuel       / MAX_FUEL,        0.0,  1.0))
        obs.damage_norm    = float(np.clip(obs.damage     / MAX_DAMAGE,      0.0,  1.0))

        # Arrays
        obs.track_sensors_norm    = np.clip(obs.track_sensors    / MAX_TRACK_DIST_M, 0.0, 1.0)
        obs.opponent_sensors_norm = np.clip(obs.opponent_sensors / MAX_OPP_DIST_M,  0.0, 1.0)
        obs.wheel_spin_vel_norm   = np.clip(obs.wheel_spin_vel   / MAX_WHEEL_VEL,   0.0, 1.0)
        obs.focus_sensors_norm    = np.clip(obs.focus_sensors    / MAX_TRACK_DIST_M, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 3 — feature engineering
    # ------------------------------------------------------------------

    def _derive_features(self, obs: SensorObservation) -> None:
        """Compute higher-level situational features."""

        # ── Speed composites ───────────────────────────────────────────
        obs.effective_speed = float(obs.speed_x * np.cos(obs.angle_rad))
        obs.speed_magnitude = float(np.sqrt(obs.speed_x ** 2 + obs.speed_y ** 2))

        # ── Track geometry ─────────────────────────────────────────────
        # Centre sensor covers straight-ahead sight-line
        obs.track_ahead = float(obs.track_sensors[TRACK_CENTER_IDX])
        # Left clearance = minimum of all sensors to the left of centre
        obs.track_left_clearance  = float(np.min(obs.track_sensors[:TRACK_CENTER_IDX]))
        # Right clearance = minimum of all sensors to the right of centre
        obs.track_right_clearance = float(np.min(obs.track_sensors[TRACK_CENTER_IDX + 1:]))

        # ── Opponent proximity ─────────────────────────────────────────
        obs.closest_opponent = float(np.min(obs.opponent_sensors))

        # ── Traction / wheel slip ──────────────────────────────────────
        # FL=0, FR=1, RL=2, RR=3
        front_spin = float(obs.wheel_spin_vel[0] + obs.wheel_spin_vel[1])
        rear_spin  = float(obs.wheel_spin_vel[2] + obs.wheel_spin_vel[3])
        obs.slip_ratio = max(0.0, rear_spin - front_spin)

        # ── Boolean situational flags ──────────────────────────────────
        obs.is_near_wall     = bool(np.min(obs.track_sensors) < self.near_wall_threshold)
        obs.is_going_forward = bool(np.cos(obs.angle_rad) > 0.0)
        obs.is_car_aligned   = bool(abs(obs.angle_rad) < self.straight_angle_threshold)
        obs.is_spinning      = bool(obs.slip_ratio > self.slip_diff_threshold)

        # ── Heuristic gear recommendation ─────────────────────────────
        obs.recommended_gear = _speed_to_gear(obs.speed_x)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(value, expected_len: int = 0, fill: float = 0.0) -> np.ndarray:
        """
        Convert *value* to a float64 numpy array, padding / truncating to
        *expected_len* if needed.
        """
        if isinstance(value, np.ndarray):
            arr = value.astype(np.float64)
        elif isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=np.float64)
        else:
            arr = np.array([float(value)], dtype=np.float64)

        if expected_len and len(arr) != expected_len:
            # Pad with fill value or truncate silently
            padded = np.full(expected_len, fill, dtype=np.float64)
            n = min(len(arr), expected_len)
            padded[:n] = arr[:n]
            return padded

        return arr


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _speed_to_gear(speed_x: float) -> int:
    """
    Return the recommended gear as an int [1..6] for a given forward speed.
    Thresholds match AlphaDriver defaults.
    """
    if speed_x > 170: return 6
    if speed_x > 140: return 5
    if speed_x > 110: return 4
    if speed_x >  80: return 3
    if speed_x >  50: return 2
    return 1


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class SensorLogger:
    """
    Writes one CSV row per TORCS timestep to a file in ``output_dir``.

    A new file is created for each episode via ``open()``; all rows are
    flushed and the file closed via ``close()``.  The writer is safe to
    construct once and reuse across many episodes.

    Output directory
    ----------------
    By default ``../results/sensors/`` relative to this file, which resolves
    to ``AiBC/results/sensors/`` in the project layout.  Pass any absolute or
    relative path to override.

    File naming
    -----------
    ``ep<NNN>_<YYYYMMDD_HHMMSS>.csv``  e.g.  ``ep001_20260418_221500.csv``

    CSV columns
    -----------
    episode, step,
    angle_rad, track_pos, speed_x, speed_y, speed_z, rpm, gear,
    fuel, damage, dist_raced, dist_from_start, cur_lap_time, last_lap_time,
    race_pos, z,
    track_0 … track_18       (19 raw distance sensors, m),
    wheel_fl, wheel_fr, wheel_rl, wheel_rr  (raw rad/s),
    angle_norm, track_pos_norm, speed_x_norm, speed_y_norm, speed_z_norm,
    rpm_norm, fuel_norm, damage_norm,
    effective_speed, speed_magnitude,
    track_ahead, track_left_clearance, track_right_clearance, closest_opponent,
    slip_ratio, recommended_gear,
    is_near_wall, is_going_forward, is_car_aligned, is_spinning

    Example
    -------
        logger = SensorLogger()           # default path
        logger.open(episode=1)
        for step in range(max_steps):
            obs = processor.process(client.S.d)
            logger.write(obs, episode=1, step=step)
        logger.close()
    """

    # Column header names — must stay in sync with _row()
    _SCALAR_HEADERS = [
        'episode', 'step',
        # raw scalars
        'angle_rad', 'track_pos', 'speed_x', 'speed_y', 'speed_z',
        'rpm', 'gear', 'fuel', 'damage',
        'dist_raced', 'dist_from_start', 'cur_lap_time', 'last_lap_time',
        'race_pos', 'z',
    ]
    _TRACK_HEADERS   = [f'track_{i}' for i in range(N_TRACK_SENSORS)]
    _WHEEL_HEADERS   = ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr']
    _NORM_HEADERS    = [
        'angle_norm', 'track_pos_norm',
        'speed_x_norm', 'speed_y_norm', 'speed_z_norm',
        'rpm_norm', 'fuel_norm', 'damage_norm',
    ]
    _DERIVED_HEADERS = [
        'effective_speed', 'speed_magnitude',
        'track_ahead', 'track_left_clearance', 'track_right_clearance',
        'closest_opponent', 'slip_ratio', 'recommended_gear',
        'is_near_wall', 'is_going_forward', 'is_car_aligned', 'is_spinning',
    ]

    HEADERS = (
        _SCALAR_HEADERS + _TRACK_HEADERS + _WHEEL_HEADERS
        + _NORM_HEADERS + _DERIVED_HEADERS
    )

    def __init__(self, output_dir: str = ''):
        if output_dir:
            self._output_dir = Path(output_dir)
        else:
            # Default: <project_root>/results/sensors/
            self._output_dir = Path(__file__).resolve().parent.parent / 'results' / 'sensors'

        self._file   = None
        self._writer = None
        self._path   = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self, episode: int) -> Path:
        """
        Open a new CSV file for *episode*.  Creates the output directory
        if it does not exist.  Closes any previously open file first.

        Returns the path of the newly created file.
        """
        if self._file is not None:
            self.close()

        self._output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f'ep{episode:03d}_{timestamp}.csv'
        self._path = self._output_dir / filename

        self._file   = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._file.flush()

        print(f'[SensorLogger] Writing to {self._path}')
        return self._path

    def write(self, obs: 'SensorObservation', episode: int, step: int) -> None:
        """
        Append one row for *obs* at (episode, step).
        Silently no-ops if the logger has not been opened.
        """
        if self._writer is None:
            return
        self._writer.writerow(self._row(obs, episode, step))
        # Flush every 100 rows so data survives a crash mid-episode
        if step % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the current CSV file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            print(f'[SensorLogger] Closed {self._path}')
        self._file   = None
        self._writer = None
        self._path   = None

    @property
    def path(self) -> Optional[Path]:
        """Path of the currently open CSV file, or None."""
        return self._path

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------

    @staticmethod
    def _row(obs: 'SensorObservation', episode: int, step: int) -> list:
        """Build the list of values for one CSV row."""
        row = [
            episode, step,
            # raw scalars
            round(obs.angle_rad,       6),
            round(obs.track_pos,       6),
            round(obs.speed_x,         4),
            round(obs.speed_y,         4),
            round(obs.speed_z,         4),
            round(obs.rpm,             2),
            obs.gear,
            round(obs.fuel,            4),
            round(obs.damage,          2),
            round(obs.dist_raced,      2),
            round(obs.dist_from_start, 2),
            round(obs.cur_lap_time,    4),
            round(obs.last_lap_time,   4),
            obs.race_pos,
            round(obs.z,               4),
        ]
        # 19 track sensors
        row += [round(v, 2) for v in obs.track_sensors.tolist()]
        # 4 wheel spin velocities
        row += [round(v, 4) for v in obs.wheel_spin_vel.tolist()]
        # normalised scalars
        row += [
            round(obs.angle_norm,     6),
            round(obs.track_pos_norm, 6),
            round(obs.speed_x_norm,   6),
            round(obs.speed_y_norm,   6),
            round(obs.speed_z_norm,   6),
            round(obs.rpm_norm,       6),
            round(obs.fuel_norm,      6),
            round(obs.damage_norm,    6),
        ]
        # derived features
        row += [
            round(obs.effective_speed,          4),
            round(obs.speed_magnitude,          4),
            round(obs.track_ahead,              2),
            round(obs.track_left_clearance,     2),
            round(obs.track_right_clearance,    2),
            round(obs.closest_opponent,         2),
            round(obs.slip_ratio,               4),
            obs.recommended_gear,
            int(obs.is_near_wall),
            int(obs.is_going_forward),
            int(obs.is_car_aligned),
            int(obs.is_spinning),
        ]
        return row


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    """
    Run  `python sensor_processor.py`  to verify the module works correctly
    without needing TORCS to be running.
    """
    import json

    # Simulate a raw packet that snakeoil3 would deliver
    mock_raw = {
        'angle':        0.05,
        'trackPos':    -0.12,
        'speedX':       87.3,
        'speedY':        1.4,
        'speedZ':        0.0,
        'rpm':        4500.0,
        'gear':            3,
        'fuel':           82.0,
        'damage':          0.0,
        'distRaced':    1234.5,
        'distFromStart': 345.6,
        'curLapTime':     28.4,
        'lastLapTime':     0.0,
        'racePos':          1,
        'z':             0.332,
        'track':        [200, 200, 180, 140, 90, 60, 45, 38, 30,
                          25,  30,  38,  45, 60, 90, 140, 180, 200, 200],
        'opponents':    [200.0] * 36,
        'wheelSpinVel': [24.2, 24.1, 24.4, 24.5],
        'focus':        [0.0, 0.0, 0.0, 0.0, 0.0],
    }

    proc = SensorProcessor()
    obs  = proc.process(mock_raw)

    print("=" * 70)
    print("  SensorProcessor — self-test")
    print("=" * 70)
    print(obs.summary())
    print()

    vec = obs.to_vector()
    print(f"  to_vector() shape      : {vec.shape}   (41 features)")
    vec_opp = obs.to_vector(include_opponents=True)
    print(f"  to_vector(opponents)   : {vec_opp.shape}  (77 features)")
    print()

    # Spot-check a few values
    assert abs(obs.speed_x_norm - 87.3 / 300.0) < 1e-5, "speed_x_norm failed"
    assert obs.track_ahead == 25.0,                       "track_ahead failed"
    assert obs.recommended_gear == 3,                     "recommended_gear failed"
    assert not obs.is_near_wall,                           "is_near_wall false positive"
    assert obs.is_car_aligned,                             "is_car_aligned failed"
    assert not obs.is_spinning,                            "is_spinning false positive"

    print("  All assertions passed [OK]")

    # ── SensorLogger test ──────────────────────────────────────────────
    print()
    print("  Testing SensorLogger...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        logger = SensorLogger(output_dir=tmp)
        csv_path = logger.open(episode=1)
        for i in range(5):
            logger.write(obs, episode=1, step=i)
        logger.close()

        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))

        n_cols = len(SensorLogger.HEADERS)
        assert len(rows) == 6,          f"Expected 6 rows (header + 5 data), got {len(rows)}"
        assert len(rows[0]) == n_cols,   f"Header has {len(rows[0])} cols, expected {n_cols}"
        assert len(rows[1]) == n_cols,   f"Data row has {len(rows[1])} cols, expected {n_cols}"
        assert rows[0][0] == 'episode',  "First header should be 'episode'"
        assert rows[1][0] == '1',        "Episode value mismatch"
        assert rows[1][1] == '0',        "Step value mismatch"

    print(f"  SensorLogger: {n_cols} columns, 5 data rows written and verified [OK]")
    print("=" * 70)

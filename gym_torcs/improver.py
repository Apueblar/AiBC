"""
improver.py — Optuna optimizer for AlphaDriver
===============================================
Score = MEAN lap time over N laps per trial (default 3).
Between laps the car is reset to start via TORCS menu navigation —
TORCS is never relaunched mid-trial.

Install:
    pip install optuna

Usage:
    python improver.py                     # 300 trials, 3 laps each
"""

import os
import time
import json
import argparse
import subprocess

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ['DISPLAY'] = ':1'

import snakeoil3_gym as snakeoil3
from car_agent import AlphaDriver, DEFAULT_PARAMS   # no duplicate driver code

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')

# ---------------------------------------------------------------------------
# TORCS GUI / process constants
# ---------------------------------------------------------------------------
_TORCS_SETTLE  = 5.0    # s — wait for TORCS window after Popen
_KEY_DELAY     = 0.25   # s — between keystrokes
_WINDOW_WAIT   = 7.0    # s — max wait for wmctrl to find the window
_WINDOW_POLL   = 0.4    # s — wmctrl poll interval
_POST_NAV_WAIT = 2.0    # s — settle time after menu navigation

# Full menu sequence from the TORCS start screen  → blue "waiting" screen
_MENU_KEYS_START = [
    'Return', 'Up', 'Up', 'Return', 'Down', 'Return',
    'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left',
    'Return', 'Return', 'Return', 'Up', 'Return',
]

# ---------------------------------------------------------------------------
# Episode settings
# ---------------------------------------------------------------------------
MAX_STEPS    = 10_000   # safety ceiling per lap  (~200 s at 50 Hz)
DNF_PENALTY  = 9_999.0  # lap-time value assigned to a DNF

# ---------------------------------------------------------------------------
# TORCS helpers
# ---------------------------------------------------------------------------

def _navigate_menu(first_run: bool = True):
    """Send the appropriate key sequence."""
    time.sleep(0.5)
    keys = _MENU_KEYS_START if first_run else []
    for key in keys:
        os.system(f"xte 'key {key}'")
        time.sleep(_KEY_DELAY)
    time.sleep(_POST_NAV_WAIT)


def launch_torcs(first_run: bool = True):
    """Kill any stale instance, start a fresh TORCS process, navigate to wait screen."""
    print(f'[improver] {"Launching" if first_run else "Restarting"} TORCS...')
    os.system('pkill -9 torcs 2>/dev/null')
    time.sleep(1.5)
    subprocess.Popen(
        'torcs -nofuel -nodamage -nolaptime',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ,
    )
    time.sleep(_TORCS_SETTLE)
    _navigate_menu(first_run=first_run)
    print('[improver] TORCS ready.')


def connect_torcs(port: int = 3001, max_wait: float = 60.0) -> 'snakeoil3.Client':
    """Retry until snakeoil3 connects or we time out."""
    deadline = time.time() + max_wait
    attempt  = 0
    while True:
        attempt += 1
        try:
            return snakeoil3.Client(p=port, vision=False)
        except Exception:
            pass
        if time.time() > deadline:
            raise RuntimeError(
                f'Cannot connect to TORCS on port {port} after {max_wait:.0f}s'
            )
        time.sleep(2.0)


def _end_episode(client):
    """Send meta=1 to tell TORCS the episode is over."""
    try:
        client.R.d['meta'] = 1
        client.respond_to_server()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single-lap runner
# ---------------------------------------------------------------------------

def run_lap(client, driver: AlphaDriver) -> tuple[float, float]:
    """
    Drive one lap from the start position.

    Returns
    -------
    lap_time : float
        Seconds for the completed lap, or 0.0 on DNF.
    dist     : float
        Metres driven (distRaced sensor).
    """
    driver.reset()
    client.MAX_STEPS = np.inf
    client.get_servers_input()

    s           = client.S.d
    lap_base    = float(s.get('lastLapTime', 0.0))
    prev_speed  = 0.0

    for step in range(MAX_STEPS):
        driver.act_raw(client.S, client.R)
        client.respond_to_server()
        client.get_servers_input()
        s = client.S.d

        speed      = float(s.get('speedX',     0.0))
        angle      = float(s.get('angle',       0.0))
        dist       = float(s.get('distRaced',   0.0))
        cur_damage = float(s.get('damage',      0.0))
        lap_now    = float(s.get('lastLapTime', 0.0))

        # -- Lap complete --------------------------------------------------
        if lap_now != lap_base and lap_now > 0.0:
            _end_episode(client)
            return lap_now, dist

        # -- Going backwards -----------------------------------------------
        if step > 100 and speed * np.cos(angle) < -1.0:
            _end_episode(client)
            return 0.0, dist

        # -- Stuck ---------------------------------------------------------
        if step > 500 and speed * np.cos(angle) < 1.0:
            _end_episode(client)
            return 0.0, dist

        # -- Off-track / wall hit detection (no-damage mode) ---------------
        track_pos = float(s.get('trackPos', 0.0))

        # 1. Car is outside the track boundaries (trackPos < -1 or > 1)
        if abs(track_pos) > 1.05:
            print(f'  [t] Off-track (trackPos={track_pos:.2f}). DNF.')
            _end_episode(client)
            return 0.0, dist

        # 2. Sudden speed crash: was going fast, now nearly stopped
        if step > 50 and prev_speed > 60 and speed < 5.0:
            print(f'  [t] Speed crash (was {prev_speed:.0f} → now {speed:.0f} km/h). DNF.')
            _end_episode(client)
            return 0.0, dist

        prev_speed = speed

    _end_episode(client)
    return 0.0, float(s.get('distRaced', 0.0))   # timeout = DNF


# ---------------------------------------------------------------------------
# Optuna search space
# ---------------------------------------------------------------------------

_prev_params: dict = {}

def _local_float(trial, name, min_abs, max_abs, variance=0.20):
    """Suggest a float within ±20% of the previous best, bounded by absolute limits."""
    center = _prev_params.get(name)
    if center is not None:
        span = max(0.05, abs(center) * variance)
        low = center - span
        high = center + span
        # Expand absolute bounds if previous center forces it
        min_abs = min(min_abs, low)
        max_abs = max(max_abs, high)
        return trial.suggest_float(name, max(min_abs, low), min(max_abs, high))
    return trial.suggest_float(name, min_abs, max_abs)

def _local_int(trial, name, min_abs, max_abs, center=None, variance=0.20):
    """Suggest an int within ±20% of the previous best, bounded by absolute limits."""
    if center is not None:
        span = max(5, int(center * variance))
        low = int(center - span)
        high = int(center + span)
        # Expand absolute bounds if previous center forces it
        min_abs = min(min_abs, low)
        max_abs = max(max_abs, high)
        return trial.suggest_int(name, max(min_abs, low), min(max_abs, high))
    return trial.suggest_int(name, min_abs, max_abs)

def _make_params(trial) -> dict:
    """Build a full params dict for one Optuna trial."""
    prev_cs = _prev_params.get('CORNER_SPEEDS', [40, 55, 80, 110, 150, 200, 280])
    prev_as = _prev_params.get('ASYM_SPEEDS',   [280, 220, 160, 110, 70, 50])

    return {
        # -- Steering (core reference formula) ----------------------------
        'K_ANGLE':            _local_float(trial, 'K_ANGLE',            2.0,  10.0),
        'K_POS':              _local_float(trial, 'K_POS',              0.03,  0.40),

        # -- Speed --------------------------------------------------------
        'TARGET_SPEED':       _local_float(trial, 'TARGET_SPEED',      180.0, 320.0),
        'STEER_SPEED_FACTOR': _local_float(trial, 'STEER_SPEED_FACTOR', 30.0, 200.0),

        # -- Forward clearance speed curve --------------------------------
        'CORNER_DISTS':  [5, 15, 30, 50, 80, 120, 200],
        'CORNER_SPEEDS': [
            _local_int(trial, 'cs_5',    25,  65, prev_cs[0]),
            _local_int(trial, 'cs_15',   35,  85, prev_cs[1]),
            _local_int(trial, 'cs_30',   50, 120, prev_cs[2]),
            _local_int(trial, 'cs_50',   70, 160, prev_cs[3]),
            _local_int(trial, 'cs_80',  100, 220, prev_cs[4]),
            _local_int(trial, 'cs_120', 140, 270, prev_cs[5]),
            _local_int(trial, 'cs_200', 200, 320, prev_cs[6]),
        ],

        # -- Asymmetry (corner anticipation) speed curve ------------------
        'ASYM_BREAKS': [0, 15, 35, 60, 90, 130],
        'ASYM_SPEEDS': [
            _local_int(trial, 'as_0',   220, 320, prev_as[0]),  # straight
            _local_int(trial, 'as_15',  150, 280, prev_as[1]),  # gentle curve
            _local_int(trial, 'as_35',   90, 200, prev_as[2]),  # medium corner
            _local_int(trial, 'as_60',   60, 140, prev_as[3]),  # sharp corner
            _local_int(trial, 'as_90',   40,  90, prev_as[4]),  # hairpin
            _local_int(trial, 'as_130',  30,  65, prev_as[5]),  # very tight
        ],

        # -- Gear (fixed) -------------------------------------------------
        'GEAR_UP':   [0,  50,  80, 110, 140, 170],
        'GEAR_DOWN': [0,   0,  35,  65,  95, 130],
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
_best_score: float = float('inf')
_cfg:        dict  = {}


def objective(trial) -> float:
    global _best_score

    params    = _make_params(trial)
    driver    = AlphaDriver(params=params)
    n_laps    = _cfg['laps']
    port      = _cfg['port']
    lap_times = []

    for lap_idx in range(n_laps):

        # -- Connect (TORCS already at waiting screen) ---------------------
        try:
            client = connect_torcs(port=port, max_wait=40)
        except RuntimeError:
            print(f'  [t{trial.number}|lap{lap_idx+1}] connect failed — relaunching TORCS')
            launch_torcs(first_run=False)
            try:
                client = connect_torcs(port=port, max_wait=50)
            except RuntimeError:
                print(f'  [t{trial.number}|lap{lap_idx+1}] relaunch failed — aborting trial')
                return float(DNF_PENALTY)

        # -- Run the lap ---------------------------------------------------
        lap_t, dist = run_lap(client, driver)

        if lap_t > 0:
            lap_times.append(lap_t)
            status = f'lap={lap_t:.2f}s'
            
            # Abort if lap is completely uncompetitive (40% slower than best)
            if _best_score != float('inf') and lap_t > _best_score * 1.40:
                print(f'  [t{trial.number}] Lap too slow ({lap_t:.2f}s > {_best_score * 1.40:.2f}s). Aborting remaining laps.')
                while len(lap_times) < n_laps:
                    lap_times.append(DNF_PENALTY)
                break
        else:
            lap_times.append(DNF_PENALTY)
            status = f'DNF'
            print(f'  trial {trial.number:>4} | lap {lap_idx+1}/{n_laps} | {status} | dist={dist:.0f}m')
            print(f'  [t{trial.number}] Car crashed/DNF. Aborting remaining laps.')
            # Fill the remaining laps with DNF_PENALTY so the mean reflects the failure
            while len(lap_times) < n_laps:
                lap_times.append(DNF_PENALTY)
            break

        print(f'  trial {trial.number:>4} | lap {lap_idx+1}/{n_laps} | '
              f'{status} | dist={dist:.0f}m')

        # -- Navigate back to waiting screen for the next lap -------------
        # (No TORCS relaunch — car resets to start via menu navigation)
        _navigate_menu(first_run=False)

    # ── Score = mean lap time (DNFs count as DNF_PENALTY) ──────────────
    score = float(np.mean(lap_times))
    completed = sum(1 for t in lap_times if t < DNF_PENALTY)
    print(f'  trial {trial.number:>4} | mean={score:.2f}s | '
          f'completed={completed}/{n_laps}')

    # -- Save immediately on new best (all laps must complete) -------------
    if score < _best_score and completed == n_laps:
        _best_score = score
        out = dict(params)
        out['best_lap_time']   = score
        out['laps_in_trial']   = n_laps
        out['trial_lap_times'] = lap_times
        with open(_PARAMS_FILE, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n  ★ NEW BEST  mean={score:.2f}s  →  {_PARAMS_FILE}\n')

    return score


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='AlphaDriver Optuna optimizer')
    p.add_argument('--trials', type=int, default=300, help='Number of Optuna trials')
    p.add_argument('--laps',   type=int, default=3,
                   help='Laps per trial for median scoring (default: 3)')
    p.add_argument('--port',   type=int, default=3001,  help='TORCS UDP port')
    return p.parse_args()


def main():
    global _cfg, _best_score, _prev_params

    args  = parse_args()
    _cfg  = {'trials': args.trials, 'laps': args.laps, 'port': args.port}

    print('=' * 58)
    print('  AlphaDriver Optimizer')
    print(f'  Trials              : {args.trials} (Local Search)')
    print(f'  Laps / trial        : {args.laps}  (scored by mean)')
    print(f'  Port                : {args.port}')
    print(f'  Best params file    : {_PARAMS_FILE}')
    print('=' * 58)

    # Load current best so we never overwrite a better result
    if os.path.exists(_PARAMS_FILE):
        with open(_PARAMS_FILE) as f:
            _prev_params = json.load(f)
        prev_lap = _prev_params.get('best_lap_time', float('inf'))
        try:
            _best_score = float(prev_lap)
            print(f'  Current best lap    : {_best_score:.2f}s')
        except (TypeError, ValueError):
            print('  Current best lap    : unknown (will overwrite on first complete run)')
    else:
        print('  No existing best_params.json — starting fresh.')
    print()

    # Warm-start from the existing best params if available
    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(
        direction  = 'minimize',
        study_name = 'alphadriver',
        sampler    = sampler,
    )

    # Launch TORCS once at the start
    launch_torcs(first_run=True)

    try:
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    except KeyboardInterrupt:
        print('\n[improver] Interrupted — saving current best and exiting.')

    try:
        best = study.best_trial
        print(f'\n[improver] Optimization complete.')
        print(f'[improver] Best mean lap   : {best.value:.2f}s')
        print(f'[improver] Params saved to : {_PARAMS_FILE}')
    except Exception:
        print('[improver] No successful trial recorded.')

    os.system('pkill -9 torcs 2>/dev/null')


if __name__ == '__main__':
    main()
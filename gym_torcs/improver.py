"""
improver.py — Optuna optimizer for AlphaDriver
===============================================
Score = MEDIAN lap time over N laps per trial (default 3).
Between laps the car is reset to start via TORCS menu navigation —
TORCS is never relaunched mid-trial.

Install:
    pip install optuna

Usage:
    python improver.py                     # 1000 trials, 3 laps each
    python improver.py --trials 200        # fewer trials
    python improver.py --laps 5            # more laps per trial (slower, stabler)
    python improver.py --port 3002
"""

import os
import sys
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
# After meta=1: race-end screen → back to blue "waiting" screen
_MENU_KEYS_CONTINUE = ['Return', 'Up', 'Up', 'Return', 'Return']

# ---------------------------------------------------------------------------
# Episode settings
# ---------------------------------------------------------------------------
MAX_STEPS    = 10_000   # safety ceiling per lap  (~200 s at 50 Hz)
DNF_PENALTY  = 9_999.0  # lap-time value assigned to a DNF

# ---------------------------------------------------------------------------
# TORCS helpers
# ---------------------------------------------------------------------------

def _find_torcs_window(timeout: float = _WINDOW_WAIT) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ['wmctrl', '-l'], stderr=subprocess.DEVNULL, env=os.environ
            ).decode()
            for line in out.splitlines():
                if 'torcs' in line.lower():
                    return line.split()[0]
        except Exception:
            pass
        time.sleep(_WINDOW_POLL)
    return None


def _navigate_menu(first_run: bool = True):
    """Focus the TORCS window (if found) and send the appropriate key sequence."""
    wid = _find_torcs_window()
    if wid:
        os.system(f'wmctrl -ia {wid}')
        time.sleep(0.5)
    else:
        print('[improver] WARNING: TORCS window not found — sending keys to focused window.')
    keys = _MENU_KEYS_START if first_run else _MENU_KEYS_CONTINUE
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
    prev_damage = float(s.get('damage', 0.0))

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

        # ── Lap complete ──────────────────────────────────────────────────
        if lap_now != lap_base and lap_now > 0.0:
            _end_episode(client)
            return lap_now, dist

        # ── Going backwards ───────────────────────────────────────────────
        if step > 100 and speed * np.cos(angle) < -1.0:
            _end_episode(client)
            return 0.0, dist

        # ── Stuck ─────────────────────────────────────────────────────────
        if step > 500 and speed * np.cos(angle) < 1.0:
            _end_episode(client)
            return 0.0, dist

        # ── Heavy collision ───────────────────────────────────────────────
        if cur_damage - prev_damage > 500:
            _end_episode(client)
            return 0.0, dist
        prev_damage = cur_damage

    _end_episode(client)
    return 0.0, float(s.get('distRaced', 0.0))   # timeout = DNF


# ---------------------------------------------------------------------------
# Optuna search space
# ---------------------------------------------------------------------------

def _make_params(trial) -> dict:
    """Build a full params dict for one Optuna trial."""
    return {
        # ── Fixed safety / physics ────────────────────────────────────────
        'TARGET_SPEED':       360.0,
        'MAX_ACCEL':          1.0,
        'PTP_DIST_THRESHOLD': 60.0,
        'TC_THRESHOLD':       50.0,
        'TC_REDUCTION':       0.20,
        'WALL_DANGER_DIST':   4.0,
        'WALL_WARN_DIST':     10.0,
        'WALL_DANGER_GAIN':   0.65,
        'WALL_WARN_GAIN':     0.20,
        'LOOKAHEAD_DISTS':    [10, 22, 40, 65, 100, 150, 200],
        'CORNER_ASYMS':       [ 0,  8, 20, 35,  52,  70],
        'GEAR_UP':            [0,  55, 100, 145, 190, 240],
        'GEAR_DOWN':          [0,   0,  40,  80, 120, 165],

        # ── Steering ─────────────────────────────────────────────────────
        'STEER_ANGLE_GAIN':  trial.suggest_float('STEER_ANGLE_GAIN',  0.35, 0.80),
        'STEER_CENTER_GAIN': trial.suggest_float('STEER_CENTER_GAIN', 0.20, 0.65),
        'STEER_LOOKAHEAD':   trial.suggest_float('STEER_LOOKAHEAD',   0.08, 0.50),
        'STEER_SPEED_DENOM': trial.suggest_float('STEER_SPEED_DENOM', 350., 750.),
        'MIN_STEER_SCALE':   trial.suggest_float('MIN_STEER_SCALE',   0.12, 0.40),
        'STEER_SMOOTH':      trial.suggest_float('STEER_SMOOTH',      0.00, 0.40),

        # ── Apex / racing line ───────────────────────────────────────────
        'APEX_OFFSET':       trial.suggest_float('APEX_OFFSET',      0.30, 0.80),
        'APEX_ASYM_THRESH':  trial.suggest_float('APEX_ASYM_THRESH', 3.0,  14.0),
        'TARGET_POS_SLEW':   trial.suggest_float('TARGET_POS_SLEW',  0.04, 0.25),

        # ── Speed slew ───────────────────────────────────────────────────
        'TARGET_SLEW_DOWN':  trial.suggest_float('TARGET_SLEW_DOWN', 2.0,  9.0),
        'TARGET_SLEW_UP':    trial.suggest_float('TARGET_SLEW_UP',   3.0, 18.0),

        # ── Braking ──────────────────────────────────────────────────────
        'BRAKE_GAIN':        trial.suggest_float('BRAKE_GAIN',       0.03, 0.16),
        'MAX_BRAKE':         trial.suggest_float('MAX_BRAKE',        0.65, 1.00),
        'BRAKE_TRIGGER':     trial.suggest_float('BRAKE_TRIGGER',    0.5,  5.0),

        # ── Lookahead speed curve (close-range tunable) ──────────────────
        'LOOKAHEAD_SPEEDS': [
            trial.suggest_int('lah_10',  25,  70),   # dist=10 m
            trial.suggest_int('lah_22',  45, 100),   # dist=22 m
            115, 155, 200, 270, 360,
        ],

        # ── Corner speed curve ───────────────────────────────────────────
        'CORNER_SPEEDS': [
            360,
            trial.suggest_int('crn_8',   190, 330),  # asym=8
            trial.suggest_int('crn_20',  100, 200),  # asym=20
            trial.suggest_int('crn_35',   55, 125),  # asym=35
            trial.suggest_int('crn_52',   38,  85),  # asym=52
            trial.suggest_int('crn_70',   30,  62),  # asym=70
        ],
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

        # ── Connect (TORCS already at waiting screen) ─────────────────────
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

        # ── Run the lap ───────────────────────────────────────────────────
        lap_t, dist = run_lap(client, driver)

        if lap_t > 0:
            lap_times.append(lap_t)
            status = f'lap={lap_t:.2f}s'
        else:
            lap_times.append(DNF_PENALTY)
            status = f'DNF'

        print(f'  trial {trial.number:>4} | lap {lap_idx+1}/{n_laps} | '
              f'{status} | dist={dist:.0f}m')

        # ── Navigate back to waiting screen for the next lap ─────────────
        # (No TORCS relaunch — car resets to start via menu navigation)
        _navigate_menu(first_run=False)

    # ── Score = median lap time (DNFs count as DNF_PENALTY) ──────────────
    score = float(np.median(lap_times))
    completed = sum(1 for t in lap_times if t < DNF_PENALTY)
    print(f'  trial {trial.number:>4} | median={score:.2f}s | '
          f'completed={completed}/{n_laps}')

    # ── Save immediately on new best (all laps must complete) ─────────────
    if score < _best_score and completed == n_laps:
        _best_score = score
        out = dict(params)
        out['best_lap_time']   = score
        out['laps_in_trial']   = n_laps
        out['trial_lap_times'] = lap_times
        with open(_PARAMS_FILE, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n  ★ NEW BEST  median={score:.2f}s  →  {_PARAMS_FILE}\n')

    return score


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='AlphaDriver Optuna optimizer')
    p.add_argument('--trials', type=int, default=1000, help='Number of Optuna trials')
    p.add_argument('--laps',   type=int, default=3,
                   help='Laps per trial for median scoring (default: 3)')
    p.add_argument('--port',   type=int, default=3001,  help='TORCS UDP port')
    return p.parse_args()


def main():
    global _cfg, _best_score

    args  = parse_args()
    _cfg  = {'trials': args.trials, 'laps': args.laps, 'port': args.port}

    print('=' * 58)
    print('  AlphaDriver Optimizer')
    print(f'  Trials              : {args.trials}')
    print(f'  Laps / trial        : {args.laps}  (scored by median)')
    print(f'  Port                : {args.port}')
    print(f'  Best params file    : {_PARAMS_FILE}')
    print('=' * 58)

    # Load current best so we never overwrite a better result
    if os.path.exists(_PARAMS_FILE):
        with open(_PARAMS_FILE) as f:
            prev = json.load(f)
        prev_lap = prev.get('best_lap_time', float('inf'))
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
        print(f'[improver] Best median lap : {best.value:.2f}s')
        print(f'[improver] Params saved to : {_PARAMS_FILE}')
    except Exception:
        print('[improver] No successful trial recorded.')

    os.system('pkill -9 torcs 2>/dev/null')


if __name__ == '__main__':
    main()
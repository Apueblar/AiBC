"""
run.py  -  Entry point for the Alpha TORCS driver
Launches TORCS, navigates its GUI menus entirely from Python (no shell
scripts, no xdotool, no wmctrl needed), then runs the AlphaDriver.

One-time dependency (if not already installed):
    pip3 install python3-xlib

Usage:
    python run.py [--vision] [--episodes N] [--steps N] [--port N] [--no-launch]
"""

import os
import sys
import time
import subprocess
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# DISPLAY must be set before importing Xlib or calling os.system()
# ---------------------------------------------------------------------------
os.environ['DISPLAY'] = ':1'

import snakeoil3_gym as snakeoil3
from car_agent import AlphaDriver
from sensor_processor import SensorProcessor, SensorLogger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Alpha TORCS driver')
    p.add_argument('--vision',    action='store_true', help='Enable camera input')
    p.add_argument('--episodes',  type=int, default=5,     help='Number of episodes')
    p.add_argument('--steps',     type=int, default=50000, help='Max steps per episode')
    p.add_argument('--port',      type=int, default=3001,  help='TORCS UDP port')
    p.add_argument('--no-launch', action='store_true',     help='Skip launching TORCS')
    return p.parse_args()


# ---------------------------------------------------------------------------
# GUI navigation – pure Python via XTEST (focus-independent)
# ---------------------------------------------------------------------------
_MENU_KEYS_START = ['Return', 'Up', 'Up', 'Return', 'Down', 'Return', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Return', 'Return', 'Return', 'Up', 'Return']
_MENU_KEYS_CONTINUE = ['Return', 'Up', 'Up', 'Return', 'Return']
_KEY_DELAY  = 0.25   # seconds between keystrokes - 0.25
_WINDOW_POLL = 0.4   # seconds between window-search retries
_WINDOW_WAIT = 15.0  # max seconds to wait for TORCS window (Time for touching things before the first strokes of episode 1 starts.)

def navigate_torcs_menu(first_run=True):
    """
    Wait for the TORCS window to appear, then send the menu keystrokes
    needed to reach the blue 'waiting for driver' screen.

    Primary path  : python3-xlib XTEST  (no external tools needed)
    Fallback path : wmctrl focus  +  xte  (if Xlib import failed)
    """
    print('[run] Waiting for TORCS window (wmctrl)...')
    deadline = time.time() + _WINDOW_WAIT
    wid = None
    while time.time() < deadline:
        try:
            out = subprocess.check_output(['wmctrl', '-l'], stderr=subprocess.DEVNULL,
                                          env=os.environ).decode()
            for line in out.splitlines():
                if 'torcs' in line.lower():
                    wid = line.split()[0]
                    break
        except Exception:
            pass
        if wid:
            break
        time.sleep(_WINDOW_POLL)

    if wid is None:
        print('[run] WARNING: could not find TORCS window via wmctrl. '
              'Keys will be sent to whatever window has focus.')
    else:
        print(f'[run] TORCS window found ({wid}). Focusing...')
        os.system(f'wmctrl -ia {wid}')
        time.sleep(0.5)

    if first_run:
        _MENU_KEYS = _MENU_KEYS_START
    else:
        _MENU_KEYS = _MENU_KEYS_CONTINUE

    for key in _MENU_KEYS:
        os.system(f"xte 'key {key}'")
        print(f'[run]   key -> {key}')
        time.sleep(_KEY_DELAY)

    print('[run] Menu navigation complete (wmctrl+xte).')

# ---------------------------------------------------------------------------
# TORCS launch
# ---------------------------------------------------------------------------
_TORCS_SETTLE = 5.0   # seconds to let TORCS open its window before we search

def launch_torcs(vision=False, new_window=True):
    """Kill any stale instance, start a fresh one, navigate the GUI."""
    print('[run] Killing any existing TORCS process...')
    os.system('pkill -9 torcs 2>/dev/null')
    time.sleep(1.5)

    flags = '-nofuel -nodamage -nolaptime'
    if vision:
        flags += ' -vision'

    print(f'[run] Starting TORCS: torcs {flags} &')
    subprocess.Popen(
        f'torcs {flags}',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ,
    )

    print(f'[run] Waiting {_TORCS_SETTLE}s for TORCS to open its window...')
    time.sleep(_TORCS_SETTLE)

    navigate_torcs_menu(first_run=new_window)

    # Small pause so TORCS finishes transitioning to the waiting screen
    time.sleep(2.0)
    print('[run] TORCS should now be at the blue waiting screen.')

# ---------------------------------------------------------------------------
# Connect to TORCS via snakeoil UDP
# ---------------------------------------------------------------------------
def connect_to_torcs(port=3001, vision=False, max_wait=90):
    print(f'[run] Connecting to TORCS on UDP port {port}...')
    deadline = time.time() + max_wait
    attempt  = 0
    while True:
        attempt += 1
        try:
            client = snakeoil3.Client(p=port, vision=vision)
            print(f'[run] Connected after {attempt} attempt(s).')
            return client
        except (SystemExit, Exception):
            pass

        if time.time() > deadline:
            raise RuntimeError(
                f'Could not connect to TORCS on port {port} within {max_wait}s. '
                'Is TORCS showing the blue waiting screen?'
            )
        print(f'[run] Not connected yet (attempt {attempt}), retrying in 2s...')
        time.sleep(2.0)

# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------
def run_episode(client, driver, processor, logger, max_steps, ep_num):
    """
    Drive until lap complete / backwards / stuck / max_steps.
    Returns (total_reward, lap_completed, lap_time, distance_raced).

    Parameters
    ----------
    processor : SensorProcessor
        Resets at episode start; processes client.S.d each step.
    logger : SensorLogger
        Opens a new CSV for this episode; writes one row per step;
        closes the file when the episode ends.
    """
    client.MAX_STEPS = np.inf
    processor.reset()           # clear any state from the previous episode
    logger.open(episode=ep_num) # AiBC/results/sensors/ep<N>_<timestamp>.csv
    client.get_servers_input()

    s = client.S.d
    obs = processor.process(s)  # first observation of the episode

    lap_time_at_start = s.get('lastLapTime', 0.0)
    prev_damage       = s.get('damage', 0.0)
    total_reward      = 0.0
    lap_completed     = False
    lap_time          = 0.0

    print(f'  [ep {ep_num}] Starting. lastLapTime baseline = {lap_time_at_start:.2f}')

    for step in range(max_steps):
        S, R = client.S, client.R
        s    = S.d

        # --- Sensor processing -----------------------------------------
        # obs contains normalised sensors + derived features every step.
        # Pass obs to your ML model here instead of (or alongside) act_raw.
        obs = processor.process(s)

        # --- Driver decision -------------------------------------------
        driver.act_raw(S, R)

        # --- Send / receive --------------------------------------------
        client.respond_to_server()
        client.get_servers_input()
        s   = client.S.d
        obs = processor.process(s)   # post-step observation

        # --- Log to CSV ------------------------------------------------
        logger.write(obs, episode=ep_num, step=step)

        speed    = obs.speed_x
        angle    = obs.angle_rad
        progress = obs.effective_speed   # already = speed_x * cos(angle)

        cur_damage  = obs.damage
        collision   = (cur_damage - prev_damage) > 0
        prev_damage = cur_damage

        reward        = progress - (10.0 if collision else 0.0)
        total_reward += reward

        if step % 500 == 0 and step > 0:
            dist = obs.dist_raced
            print(f'  [ep {ep_num} | step {step:>6}]  {obs.summary()}  '
                  f'dist={dist:7.1f}m  reward={total_reward:8.1f}')

        current_lap_time = obs.last_lap_time
        if current_lap_time != lap_time_at_start and current_lap_time > 0.0:
            lap_time      = current_lap_time
            lap_completed = True
            print(f'\n  *** LAP COMPLETE at step {step} ***')
            print(f'  Lap time : {lap_time:.2f} s')
            print(f'  Distance : {obs.dist_raced:.1f} m')
            print(f'  Reward   : {total_reward:.2f}\n')
            client.R.d['meta'] = 1
            client.respond_to_server()
            logger.close()
            break

        # Termination checks using obs flags (cleaner than raw dict access)
        if not obs.is_going_forward:
            print(f'  [ep {ep_num} | step {step}] Driving backwards - ending episode.')
            client.R.d['meta'] = 1
            client.respond_to_server()
            logger.close()
            break

        if step > 500 and progress < 1.0:
            print(f'  [ep {ep_num} | step {step}] Stuck (progress={progress:.2f}) - ending episode.')
            client.R.d['meta'] = 1
            client.respond_to_server()
            logger.close()
            break

    else:
        print(f'  [ep {ep_num}] Reached max steps ({max_steps}) without completing a lap.')
        logger.close()

    return total_reward, lap_completed, lap_time, obs.dist_raced

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args      = parse_args()
    driver    = AlphaDriver()
    processor = SensorProcessor()   # shared across episodes; reset() each episode
    logger    = SensorLogger()      # writes AiBC/results/sensors/ep<N>_<timestamp>.csv
    print('[run] AlphaDriver initialised.')
    print('[run] SensorProcessor initialised.')
    print(f'[run] SensorLogger output dir: {logger._output_dir}')

    if not args.no_launch:
        launch_torcs(vision=args.vision, new_window=True)

    laps_completed = 0

    for ep in range(args.episodes):
        print(f'\n{"="*55}')
        print(f'  Episode {ep + 1} / {args.episodes}')
        print(f'{"="*55}')

        # Relaunch TORCS every 3rd episode to avoid memory leak
        if ep > 0 and ep % 3 == 0:
            print('[run] Relaunching TORCS to avoid memory leak...')
            launch_torcs(vision=args.vision, new_window=ep % 3 == 0)

        try:
            client = connect_to_torcs(port=args.port, vision=args.vision)
        except RuntimeError as e:
            print(f'[run] Connection failed: {e}')
            print('[run] Attempting TORCS relaunch and retry...')
            launch_torcs(vision=args.vision, new_window=ep % 3 == 0)
            client = connect_to_torcs(port=args.port, vision=args.vision)

        total_reward, lap_done, lap_time, distance = run_episode(
            client, driver, processor, logger, max_steps=args.steps, ep_num=ep + 1
        )

        if lap_done:
            laps_completed += 1

        print(f'\n  Episode {ep + 1} summary:')
        print(f'    Lap completed : {"YES  ✓" if lap_done else "No"}')
        if lap_done:
            print(f'    Lap time      : {lap_time:.2f} s')
        print(f'    Distance      : {distance:.1f} m')
        print(f'    Total reward  : {total_reward:.2f}')

        try:
            client.R.d['meta'] = 1
            client.respond_to_server()
        except Exception:
            pass

    print(f'\n{"="*55}')
    print(f'  Run complete. Laps finished: {laps_completed} / {args.episodes}')
    print(f'{"="*55}')
    print('[run] Shutting down TORCS.')
    os.system('pkill -9 torcs 2>/dev/null')

if __name__ == '__main__':
    main()
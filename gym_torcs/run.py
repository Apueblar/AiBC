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

# ---------------------------------------------------------------------------
# Xlib – used for focus-independent key injection via the XTEST extension
# ---------------------------------------------------------------------------
try:
    from Xlib import X, XK, display as xdisplay
    from Xlib.ext import xtest as xtest_ext
    _XLIB_OK = True
except ImportError:
    _XLIB_OK = False

import snakeoil3_gym as snakeoil3
from car_agent import AlphaDriver


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
_MENU_KEYS = ['Return', 'Up', 'Up', 'Return', 'Return']
_KEY_DELAY  = 0.25   # seconds between keystrokes
_WINDOW_POLL = 0.4   # seconds between window-search retries
_WINDOW_WAIT = 25.0  # max seconds to wait for TORCS window


def _find_torcs_window(dpy):
    """Recursively walk the X11 window tree; return first window named 'torcs'."""
    def _search(win):
        try:
            name = win.get_wm_name() or ''
            if 'torcs' in name.lower():
                return win
        except Exception:
            pass
        try:
            for child in win.query_tree().children:
                found = _search(child)
                if found:
                    return found
        except Exception:
            pass
        return None
    return _search(dpy.screen().root)


def _send_key_xlib(dpy, keysym_name):
    """Inject a key press+release at the X server level (ignores focus)."""
    keysym  = XK.string_to_keysym(keysym_name)
    keycode = dpy.keysym_to_keycode(keysym)
    xtest_ext.fake_input(dpy, X.KeyPress,   keycode)
    dpy.sync()
    time.sleep(0.05)
    xtest_ext.fake_input(dpy, X.KeyRelease, keycode)
    dpy.sync()


def navigate_torcs_menu():
    """
    Wait for the TORCS window to appear, then send the menu keystrokes
    needed to reach the blue 'waiting for driver' screen.

    Primary path  : python3-xlib XTEST  (no external tools needed)
    Fallback path : wmctrl focus  +  xte  (if Xlib import failed)
    """
    # ── Primary: python3-xlib ──────────────────────────────────────────────
    if _XLIB_OK:
        try:
            dpy = xdisplay.Display()
            if not dpy.has_extension('XTEST'):
                raise RuntimeError('XTEST extension not available')

            print('[run] Waiting for TORCS window (Xlib)...')
            deadline = time.time() + _WINDOW_WAIT
            win = None
            while time.time() < deadline:
                win = _find_torcs_window(dpy)
                if win:
                    break
                time.sleep(_WINDOW_POLL)

            if win is None:
                raise RuntimeError(f'TORCS window not found within {_WINDOW_WAIT}s')

            print(f'[run] TORCS window found (0x{win.id:08x}). Sending menu keys...')
            time.sleep(0.3)   # let the menu fully render

            for key in _MENU_KEYS:
                _send_key_xlib(dpy, key)
                print(f'[run]   key -> {key}')
                time.sleep(_KEY_DELAY)

            dpy.close()
            print('[run] Menu navigation complete (Xlib).')
            return

        except Exception as e:
            print(f'[run] Xlib path failed ({e}), trying wmctrl fallback...')

    # ── Fallback: wmctrl + xte ─────────────────────────────────────────────
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

    for key in _MENU_KEYS:
        os.system(f"xte 'key {key}'")
        print(f'[run]   key -> {key}')
        time.sleep(_KEY_DELAY)

    print('[run] Menu navigation complete (wmctrl+xte).')


# ---------------------------------------------------------------------------
# TORCS launch
# ---------------------------------------------------------------------------
_TORCS_SETTLE = 5.0   # seconds to let TORCS open its window before we search

def launch_torcs(vision=False):
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

    navigate_torcs_menu()

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
def run_episode(client, driver, max_steps, ep_num):
    """
    Drive until lap complete / (off-track - removed) / backwards / stuck / max_steps.
    Returns (total_reward, lap_completed, lap_time, distance_raced).
    """
    client.MAX_STEPS = np.inf
    client.get_servers_input()

    s = client.S.d
    lap_time_at_start = s.get('lastLapTime', 0.0)
    prev_damage       = s.get('damage', 0.0)
    total_reward      = 0.0
    lap_completed     = False
    lap_time          = 0.0

    print(f'  [ep {ep_num}] Starting. lastLapTime baseline = {lap_time_at_start:.2f}')

    for step in range(max_steps):
        S, R = client.S, client.R
        s    = S.d

        driver.act_raw(S, R)

        client.respond_to_server()
        client.get_servers_input()
        s = client.S.d

        speed    = s['speedX']
        angle    = s['angle']
        progress = speed * np.cos(angle)

        cur_damage  = s.get('damage', 0.0)
        collision   = (cur_damage - prev_damage) > 0
        prev_damage = cur_damage

        reward        = progress - (10.0 if collision else 0.0)
        total_reward += reward

        if step % 500 == 0 and step > 0:
            dist = s.get('distRaced', 0.0)
            print(f'  [ep {ep_num} | step {step:>6}]  '
                  f'speed={speed:5.1f} km/h  dist={dist:7.1f} m  '
                  f'reward={total_reward:8.1f}')

        current_lap_time = s.get('lastLapTime', 0.0)
        if current_lap_time != lap_time_at_start and current_lap_time > 0.0:
            lap_time      = current_lap_time
            lap_completed = True
            dist          = s.get('distRaced', 0.0)
            print(f'\n  *** LAP COMPLETE at step {step} ***')
            print(f'  Lap time : {lap_time:.2f} s')
            print(f'  Distance : {dist:.1f} m')
            print(f'  Reward   : {total_reward:.2f}\n')
            client.R.d['meta'] = 1
            client.respond_to_server()
            break

        track = np.array(s['track'])

        # if track.min() < 0:
        #     print(f'  [ep {ep_num} | step {step}] Off track – ending episode.')
        #     client.R.d['meta'] = 1
        #     client.respond_to_server()
        #     break

        if np.cos(angle) < 0:
            print(f'  [ep {ep_num} | step {step}] Driving backwards – ending episode.')
            client.R.d['meta'] = 1
            client.respond_to_server()
            break

        if step > 500 and progress < 1.0:
            print(f'  [ep {ep_num} | step {step}] Stuck (progress={progress:.2f}) – ending episode.')
            client.R.d['meta'] = 1
            client.respond_to_server()
            break

    else:
        print(f'  [ep {ep_num}] Reached max steps ({max_steps}) without completing a lap.')

    dist = s.get('distRaced', 0.0)
    return total_reward, lap_completed, lap_time, dist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Warn early if python3-xlib is missing so the user can fix it once
    if not _XLIB_OK:
        print('[run] WARNING: python3-xlib not found. GUI navigation will use '
              'wmctrl+xte as fallback (may be less reliable).')
        print('[run]   To install: pip3 install python3-xlib')

    args   = parse_args()
    driver = AlphaDriver()
    print('[run] AlphaDriver initialised.')

    if not args.no_launch:
        launch_torcs(vision=args.vision)

    laps_completed = 0

    for ep in range(args.episodes):
        print(f'\n{"="*55}')
        print(f'  Episode {ep + 1} / {args.episodes}')
        print(f'{"="*55}')

        # Relaunch TORCS every 3rd episode to avoid memory leak
        if ep > 0 and ep % 3 == 0:
            print('[run] Relaunching TORCS to avoid memory leak...')
            launch_torcs(vision=args.vision)

        try:
            client = connect_to_torcs(port=args.port, vision=args.vision)
        except RuntimeError as e:
            print(f'[run] Connection failed: {e}')
            print('[run] Attempting TORCS relaunch and retry...')
            launch_torcs(vision=args.vision)
            client = connect_to_torcs(port=args.port, vision=args.vision)

        total_reward, lap_done, lap_time, distance = run_episode(
            client, driver, max_steps=args.steps, ep_num=ep + 1
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
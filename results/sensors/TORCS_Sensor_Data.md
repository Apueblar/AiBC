# TORCS Sensor Data Dictionary

This document explains the data logged during TORCS (The Open Racing Car Simulator) AI training episodes, specifically the CSV files generated in the `results/sensors/` directory (e.g., `ep001_20260420_153537.csv`).

Each row in these CSV files represents a single time step (or "snapshot") of the simulation. The data describes everything the AI agent "feels", "sees", and its internal state at that exact millisecond. 

---

## 1. Context & Timing Variables
These variables keep track of the simulation's progress over time.

* **`episode`**: The current training or evaluation run number.
* **`step`**: The discrete time step within the current episode.
* **`cur_lap_time`**: Time elapsed since the start of the current lap (in seconds).
* **`last_lap_time`**: Total time taken to complete the previous lap (in seconds).

**Why it matters:** Essential for tracking the agent's lifespan in an episode, measuring improvement over generations, and evaluating racing performance (lap times).

---

## 2. Spatial Position & Orientation
These parameters define exactly where the vehicle is relative to the track and how it is oriented.

* **`angle_rad`**: The angle of the car relative to the longitudinal track axis (in radians). 0 means perfectly straight along the track.
* **`track_pos`**: Lateral position of the car on the track. `0` is the exact center. `-1` is the left edge, and `+1` is the right edge.
* **`dist_raced`**: Total distance the car has driven overall (meters).
* **`dist_from_start`**: The car's position measured along the track's center line from the start/finish line (meters).
* **`race_pos`**: Current rank/position against opponents (e.g., 1st, 2nd).
* **`z`**: The vertical distance/height of the car's center of mass above the road surface (meters).

**Why it matters:** The AI uses `track_pos` to stay on the road, `angle_rad` to steer effectively down straights and curves, and distance measures to track long-term progress.

---

## 3. Physical State & Velocity
Core physics parameters defining how the car is moving.

* **`speed_x`**: Forward velocity of the car (km/h).
* **`speed_y`**: Lateral (sideways) velocity of the car (km/h). Positive is rightward.
* **`speed_z`**: Vertical velocity (km/h). Usually small unless jumping or crashing.
* **`rpm`**: The engine's Revolutions Per Minute.
* **`gear`**: The currently engaged gear (`-1` = Reverse, `0` = Neutral, `1-6` = Drive).

**Why it matters:** The agent needs to know its speed to brake for corners. `rpm` and `gear` are essential if the AI is responsible for manual shifting to maintain optimal acceleration.

---

## 4. Vehicle Status & Vitals
* **`fuel`**: Amount of fuel remaining.
* **`damage`**: Cumulative damage points the car has taken from crashing or grinding against walls.
* **`wheel_fl`, `wheel_fr`, `wheel_rl`, `wheel_rr`**: The rotational velocity (spin rate) of each of the four wheels in radians per second (Front-Left, Front-Right, Rear-Left, Rear-Right).

**Why it matters:** Managing fuel and minimizing damage are critical for finishing a race. Wheel spin rates are crucial for detecting when the car is losing traction (e.g., during burnout or sliding outward on a corner).

---

## 5. Track Rangefinder Sensors (`track_0` to `track_18`)
19 simulated laser rangefinders (like LIDAR) mounted on the front of the car. They fan out from -45 degrees to +45 degrees in front of the vehicle. 

* **`track_0`**: Points 45 degrees to one side.
* **`track_9`**: Points exactly 0 degrees (dead straight ahead).
* **`track_18`**: Points 45 degrees to the opposite side.
* The remaining sensors are spread out at intermediate closing angles (19°, 12°, 7°, etc.).

**Value meaning:** The distance in meters (up to 200m) to the physical edge of the track.

**Why it matters:** This is the primary "vision" system for the car. It allows the AI to "see" upcoming corners (e.g., if standard straight-ahead sensors suddenly read low distances while the right-side sensors read high, the track is curving right).

---

## 6. Normalised ML Features (`*_norm`)
Several of the raw variables mentioned above are duplicated with a `_norm` suffix (e.g., `speed_x_norm`, `rpm_norm`, `track_0_norm`).

**What it means:** These are the raw values scaled down to a specific mathematical range, usually `[0, 1]` or `[-1, 1]`. For example, if max speed is 300 km/h, a `speed_x` of 150 km/h becomes a `speed_x_norm` of `0.5`.

**Why it matters:** Machine Learning models (like Neural Networks) train significantly faster and more reliably when their inputs are kept within small, standardized bounds.

---

## 7. Engineered / Derived Features
These are not raw sensors from the game, but mathematically derived signals calculated in `sensor_processor.py` to give the AI agent immediate "situational awareness" without forcing it to learn the complex math first.

* **`track_ahead`**: Value of exactly the center rangefinder (`track_9`).
* **`track_left_clearance` / `track_right_clearance`**: The minimum distance of all sensors on the left/right side. Tells the AI instantly if there is an imminent wall on that side.
* **`slip_ratio`**: The difference in speed between the rear wheels and front wheels.
* **`effective_speed`**: `speed_x * cos(angle_rad)`. Measures actual track-axis progression instead of just forward momentum (which might be pointing towards a wall).
* **Boolean Flags (`is_near_wall`, `is_going_forward`, `is_car_aligned`, `is_spinning`)**: Simple `0` (False) or `1` (True) flags indicating specific critical conditions based on thresholds.
* **`recommended_gear`**: A hardcoded heuristic that suggests the optimal gear for the current speed.

**Why it matters:** These engineered features provide strong, pre-calculated hints to the AI. For instance, instead of the Neural Network having to figure out how to compare 4 wheel speeds to detect traction loss, it can simply look at the `is_spinning` flag and react immediately by easing off the throttle.

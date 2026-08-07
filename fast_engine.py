import json
import sys
import numpy as np
from numba import njit
import multiprocessing as mp
# =====================================================================
# STEP 2: HIGH-SPEED JIT PHYSICS HELPERS
# =====================================================================

@njit(fastmath=True)
def fast_euclidean_distance(x1, y1, x2, y2):
    """
    Lightning-fast point-to-point distance for trajectory tracking.
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

@njit(fastmath=True)
def check_swept_collision(p1_x, p1_y, v1_x, v1_y, p2_x, p2_y, v2_x, v2_y, radius, dt):
    """
    Continuous swept-pair collision check between two entities.
    Runs at raw C-speed on the CPU.
    """
    dx = p2_x - p1_x
    dy = p2_y - p1_y
    dvx = v2_x - v1_x
    dvy = v2_y - v1_y
    
    a = dvx * dvx + dvy * dvy
    if a == 0.0:
        return False
        
    b = 2.0 * (dx * dvx + dy * dvy)
    c = (dx * dx + dy * dy) - (radius * radius * 4.0)
    
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False
        
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    return 0.0 <= t <= dt

@njit(fastmath=True)
def calculate_logarithmic_speed(base_speed, thermal_energy, decay_factor):
    """
    Applies logarithmic speed scaling based on thermal load.
    """
    return base_speed / (1.0 + decay_factor * np.log1p(thermal_energy))
@njit(fastmath=True)
def check_swept_collision(p1, v1, p2, v2, radius, dt):
    """
    JIT-compiled continuous collision check.
    Runs in C-speed directly on CPU cores.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dvx = v2[0] - v1[0]
    dvy = v2[1] - v1[1]
    
    a = dvx*dvx + dvy*dvy
    if a == 0.0:
        return False
        
    b = 2.0 * (dx*dvx + dy*dvy)
    c = (dx*dx + dy*dy) - (radius * radius * 4.0)
    
    discriminant = b*b - 4.0*a*c
    if discriminant < 0.0:
        return False
        
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    return 0.0 <= t <= dt

def evaluate_early_surrender(state, step):
    """
    Evaluates if the match is mathematically decided to save CPU time.
    Condition: After step 100, if a player owns > 75% of planets AND has > 3x ships.
    """
    if step < 100:
        return None  # Too early to call

    p0_planets = p1_planets = 0
    p0_ships = p1_ships = 0
    total_planets = len(state.get("planets", []))
    
    if total_planets == 0:
        return None

    # 1. Tally planets and garrisoned ships
    for planet in state.get("planets", []):
        owner = planet[1]  # Assuming format: [id, owner, ships, x, y, ...]
        ships = planet[2]
        if owner == 0:
            p0_planets += 1
            p0_ships += ships
        elif owner == 1:
            p1_planets += 1
            p1_ships += ships

    # 2. Add active fleet ships to the total tally
    for fleet in state.get("fleets", []):
        owner = fleet[1]  # Assuming format: [id, owner, ships, source, target, ...]
        ships = fleet[2]
        if owner == 0:
            p0_ships += ships
        elif owner == 1:
            p1_ships += ships

    # 3. Check Player 0 dominance
    if p0_planets > (0.75 * total_planets) and p0_ships > (3 * p1_ships):
        return 0  # Player 0 wins early

    # 4. Check Player 1 dominance
    if p1_planets > (0.75 * total_planets) and p1_ships > (3 * p0_ships):
        return 1  # Player 1 wins early

    return None  # Match is still competitive, continue simulation


def run_fast_match(bot_0_logic, bot_1_logic, initial_state):
    """
    Stripped-down headless engine loop focused purely on speed.
    """
    state = initial_state
    
    for step in range(500):
        # [Engine Physics & Movement Resolution Logic Goes Here]
        # state = resolve_turn(state, bot_0_logic, bot_1_logic, step)
        
        # The 70% Speed Hack: Early Surrender Check
        winner = evaluate_early_surrender(state, step)
        if winner is not None:
            return winner  # Instantly break the 500-step loop and return the result

    # Standard tie-breaker or final tally if the match reaches step 500
    return evaluate_early_surrender(state, 500) or -1


# ... [Your previous early surrender and Numba code is up here] ...

def worker_run_match(match_args):
    """
    Unpacks arguments and runs a single match.
    Required because multiprocessing.map needs a single iterable argument.
    """
    bot_0, bot_1, initial_state = match_args
    # Assuming run_fast_match is the engine loop you made in Step 1
    return run_fast_match(bot_0, bot_1, initial_state)

def run_parallel_tournament(match_list, num_cores=4):
    """
    Distributes thousands of matches across all available CPU cores.
    match_list format: [(bot_A, bot_B, state1), (bot_A, bot_C, state2), ...]
    """
    # Create a pool of workers matching the number of CPU cores
    with mp.Pool(processes=num_cores) as pool:
        # map() automatically chunks the match_list and feeds it to the idle cores
        results = pool.map(worker_run_match, match_list)
    
    return results

import time
import os

def save_checkpoint(generation, leo_dna, filename="/kaggle/working/leo_checkpoint.json"):
    """
    Saves Leo's current Macro-Brain parameters to Kaggle's output directory.
    """
    checkpoint_data = {
        "generation": generation,
        "dna": leo_dna,
        "timestamp": time.time()
    }
    
    # Write to a temporary file first, then rename to prevent corruption 
    temp_filename = filename + ".tmp"
    with open(temp_filename, 'w') as f:
        json.dump(checkpoint_data, f)
    
    os.replace(temp_filename, filename)
    print(f"[SAVE] Generation {generation} safely written to {filename}")

def load_checkpoint(filename="/kaggle/working/leo_checkpoint.json"):
    """
    Loads Leo's DNA if the script is restarted.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"[LOAD] Resuming from generation {data['generation']}")
        return data['generation'], data['dna']
    return 0, None  # Start from scratch if no checkpoint exists
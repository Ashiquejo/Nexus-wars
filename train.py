import time
import os
import json
import random
from fast_engine import load_checkpoint, save_checkpoint, run_parallel_tournament

# ... [Keep your MAX_RUNTIME_SECONDS and train_leo() function exactly as is] ...
import numpy as np
import random

# Define the number of parameters Leo uses to make decisions
DNA_SIZE = 50 

def initialize_leo_dna():
    """
    Creates the very first generation of Leo's brain.
    Initialized as a numpy array of random weights between -1.0 and 1.0.
    """
    # Using list conversion so it easily serializes to JSON for our checkpoints
    return np.random.uniform(-1.0, 1.0, DNA_SIZE).tolist()

def mutate_and_evolve(match_results, current_dna, mutation_rate=0.1, mutation_scale=0.2):
    """
    Evaluates how well Leo did in the tournament and mutates the DNA.
    """
    # 1. Calculate Fitness (Win Rate)
    # Assuming match_results is a list of winners (0 for Leo, 1 for Opponent, -1 for Tie)
    wins = match_results.count(0)
    ties = match_results.count(-1)
    total_matches = len(match_results)
    
    if total_matches == 0:
        return current_dna
        
    win_rate = (wins + (ties * 0.5)) / total_matches
    print(f"[EVOLUTION] Generation Win Rate: {win_rate * 100:.2f}%")

    # 2. Evolutionary Strategy
    dna_array = np.array(current_dna)
    
    # If Leo performed poorly (e.g., win rate < 40%), we mutate aggressively
    if win_rate < 0.40:
        print("-> Poor performance. Aggressive mutation applied.")
        mutation_scale = 0.5  # Wider variance
        mutation_rate = 0.3   # More genes mutate
    # If Leo performed well, we only apply micro-tweaks to refine the strategy
    elif win_rate >= 0.60:
        print("-> Good performance. Micro-mutations applied.")
        mutation_scale = 0.05
        mutation_rate = 0.05
        
    # 3. Apply Gaussian Noise (Mutation)
    # Create a mask of which genes will mutate based on mutation_rate
    mutation_mask = np.random.rand(DNA_SIZE) < mutation_rate
    
    # Generate random noise using a normal distribution
    # Equation: \Delta w = \mathcal{N}(0, \sigma^2)
    noise = np.random.normal(loc=0.0, scale=mutation_scale, size=DNA_SIZE)
    
    # Apply the noise only to the selected genes
    dna_array[mutation_mask] += noise[mutation_mask]
    
    # Clip weights to prevent them from exploding to infinity
    dna_array = np.clip(dna_array, -5.0, 5.0)
    
    return dna_array.tolist()

def generate_matchups(leo_dna, generations_to_mix=5, num_opponents=30, matches_per_opponent=4):
    """
    Creates a diverse list of matches for Leo (Player 0).
    """
    match_list = []
    
    # 1. Load the initial board state once to save I/O time
    if os.path.exists("initial_state.json"):
        with open("initial_state.json", 'r') as f:
            initial_state = json.load(f)
    else:
        # Ensure you define this fallback!
        initial_state = {} 

    # 2. Add Past Leo Generations
    if os.path.exists("/kaggle/working/leo_checkpoints"):
        checkpoint_files = sorted(
            [f for f in os.listdir("/kaggle/working/leo_checkpoints") if f.endswith(".json")],
            key=lambda x: int(x.split('_')[1].split('.')[0]) 
        )
        
        recent_checkpoints = checkpoint_files[-generations_to_mix:]
        
        for checkpoint in recent_checkpoints:
            full_path = f"/kaggle/working/leo_checkpoints/{checkpoint}"
            with open(full_path, 'r') as f:
                data = json.load(f)
                past_dna = data['dna']
                
                for _ in range(matches_per_opponent // 2):
                    # CORRECTED FORMAT: (Bot 0 DNA, Bot 1 DNA, Board State)
                    match_list.append((leo_dna, past_dna, initial_state))

    # 3. Add Baseline Bots (Hades & Random)
    for _ in range(num_opponents):
        for _ in range(matches_per_opponent):
            match_list.append((leo_dna, "random_bot", initial_state))
            
            # If Hades is ready, you can swap it in here
            # match_list.append((leo_dna, "hades_bot", initial_state)) 

    # 4. Shuffle to randomize opponent order
    random.shuffle(match_list)
    return match_list[:1200]
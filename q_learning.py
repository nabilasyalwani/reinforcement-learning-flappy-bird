import numpy as np
import random
import sys
sys.path.append("game/")
from game.wrapped_flappy_bird import GameState

ACTIONS = 2  # 0: do nothing, 1: flap
NUM_BINS = 10

# Fungsi diskretisasi
def get_features(game_state): 
    bird_y = game_state.get_bird_y()          
    pipe_x = game_state.get_next_pipe_x()    
    pipe_gap_y = game_state.get_next_pipe_gap_y() 
    return (bird_y, pipe_x, pipe_gap_y)

# Inisialisasi Q-table
Q = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, ACTIONS))

# Hyperparameter
ALPHA = 0.7       
GAMMA = 0.95     
INITIAL_EPSILON = 0.01    
EXPLORE = 100000
FINAL_EPSILON = 0.01
EPISODES = 10000

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(ACTIONS)
    else:
        return np.argmax(Q[state])

def train():
    epsilon = INITIAL_EPSILON
    game_state = GameState()

    for episode in range(EPISODES):
        # Reset game
        game_state = GameState()  
        a_t = np.zeros(ACTIONS)
        a_t[0] = 1  

        # Start game to get first state
        _, _, done = game_state.frame_step(a_t)

        state = get_features(game_state)
        total_reward = 0

        while not done:
            action = choose_action(state, epsilon)

            a_t = np.zeros(ACTIONS)
            a_t[action] = 1

            _, reward, done = game_state.frame_step(a_t)
            next_state = get_features(game_state)
            
            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            if done:
                td_target = reward
            else:
                td_target = reward + GAMMA * Q[next_state][best_next_action]

            td_error = td_target - Q[state][action]
            Q[state][action] += ALPHA * td_error

            state = next_state
            total_reward += reward

        print(f"Episode {episode} - Total reward: {total_reward} - Epsilon: {epsilon:.4f}")

if __name__ == "__main__":
    train()

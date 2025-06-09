import time
import gymnasium as gym
import metaworld
import numpy as np

# === CONFIGURATION ===
ENV_NAME = 'custom-two-balls'  # Name of the environment to visualize
NUM_EPISODES = 10
STEPS_PER_EPISODE = 250  # Number of steps to keep each episode running

# === WRAPPER (Required for correct environment setup) ===
# This wrapper handles setting a new random task for the environment
# at the beginning of each episode.
class MetaworldWrapper(gym.Wrapper):
    """
    Wrapper to set a new random task from a Metaworld benchmark for each episode.
    """
    def __init__(self, env, benchmark):
        super().__init__(env)
        self.benchmark = benchmark
        self.tasks = iter(self.benchmark.train_tasks)

    def reset(self, **kwargs):
        """Resets the environment and sets a new task."""
        try:
            task = next(self.tasks)
        except StopIteration:
            # Reshuffle and start over if all tasks have been used
            self.tasks = iter(self.benchmark.train_tasks)
            task = next(self.tasks)

        self.env.set_task(task)
        return super().reset(**kwargs)

# --- Environment Creation Function ---
def make_metaworld_env(env_name, render_mode=None):
    """
    Creates and wraps the Metaworld environment.
    """
    ml1 = metaworld.ML1(env_name)
    env_class = ml1.train_classes[env_name]

    env = env_class(render_mode=render_mode)
    env = MetaworldWrapper(env, ml1)
    return env

def main():
    # --- Create Environment ---
    env = make_metaworld_env(ENV_NAME, render_mode="human")

    # --- Define a Zero-Action ---
    # This action will command the robot to apply no force, so it remains still.
    zero_action = np.zeros(env.action_space.shape)

    # --- Run Visualization Loop ---
    for episode in range(NUM_EPISODES):
        print(f"--- Episode {episode + 1} ---")
        obs, _ = env.reset()
        
        for step in range(STEPS_PER_EPISODE):
            # Render the current frame
            env.render()

            # Take a step with the zero-action
            obs, reward, terminated, truncated, info = env.step(zero_action)

            # Pause for a fraction of a second to make the rendering smooth
            time.sleep(0.02)

            if terminated or truncated:
                break
            
            print("OBS:", obs)

    # --- Clean up ---
    env.close()
    print("\nVisualization finished.")

if __name__ == "__main__":
    main()
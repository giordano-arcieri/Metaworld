import os
import sys
import time  # Import the time module
import gymnasium as gym
import metaworld
from stable_baselines3 import PPO

# === Configuration ===
MODEL_PATH = f"ppo_models/pick_place/models/{sys.argv[1:][0]}.zip"
ENV_NAME = "reach-v3"
NUM_EPISODES = 10

# === WRAPPER (Required for correct environment setup) ===
# This wrapper is essential for using Metaworld, as it handles setting a
# new random task for the environment when reset() is called.
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
            self.tasks = iter(self.benchmark.train_tasks)
            task = next(self.tasks)

        self.env.set_task(task)
        return super().reset(**kwargs)

# --- Environment Creation Function ---
def make_metaworld_env(env_name, render_mode=None):
    """
    Creates and wraps the Metaworld environment for evaluation.
    """
    mt1 = metaworld.MT1(env_name)
    env_class = mt1.train_classes[env_name]

    # Get all available training tasks
    tasks = mt1.train_tasks

    # Instantiate the environment
    env = env_class(render_mode=render_mode)

    # Set partially observable to False
    # env._partially_observable = False

    env.set_task(tasks[0])
    return env

# === MAIN EVALUATION SCRIPT ===
def main():
    # --- Load Model and Create Environment ---
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please update the path.")

    model = PPO.load(MODEL_PATH)
    env = make_metaworld_env(ENV_NAME, render_mode="human")

    print(f"Running {NUM_EPISODES} evaluation episodes for {ENV_NAME}...")

    # --- Run Evaluation Loop ---
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        step = 0

        while not terminated:
            # Render
            env.render()

            # Use the model to predict the best action
            action, _ = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Printing at every step can be overwhelming.
            # Consider printing only periodically, for example, every 1000 steps.
            if step % 100 == 0:
                print(f"\n--- Step: {step} ---")
                print(f"  - Reward: {reward:.5f}")
                # The state vector can be large, so we print its shape and a small slice.
                print(f"  - State (Observation): {obs}")
                print("--------------------------")

            # Pause for 50 milliseconds to make the rendering viewable
            time.sleep(0.05)

            # End the episode if truncated
            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Success = {info.get('success', 0.0)}")

    # --- Clean up ---
    env.close()
    print("\nEvaluation finished.")

if __name__ == "__main__":
    main()
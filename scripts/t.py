import os
import random

import gymnasium as gym
import metaworld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class MetaworldWrapper(gym.Wrapper):
    """
    Wrapper for Metaworld environments to handle the task setting for each episode.

    The stable_baselines3 agent will automatically call env.reset() when an
    episode ends. This wrapper intercepts that call and sets a new random
    task from the benchmark before calling the actual environment's reset.
    """
    def __init__(self, env, benchmark):
        super().__init__(env)
        self.benchmark = benchmark
        self.tasks = iter(self.benchmark.train_tasks)

    def reset(self, **kwargs):
        """
        Reset the environment and set a new task.
        """
        try:
            task = next(self.tasks)
        except StopIteration:
            # If we've used all tasks, reshuffle them and start over
            self.tasks = iter(self.benchmark.train_tasks)
            task = next(self.tasks)
        
        self.env.set_task(task)
        return super().reset(**kwargs)

def main():
    """
    Train a PPO agent on the Metaworld pick-place-v2 environment.
    """
    # --- 1. Create the Metaworld Environment and Wrapper ---
    print("Initializing Metaworld environment...")
    ml1 = metaworld.ML1('pick-place-v3')
    env_class = ml1.train_classes['pick-place-v3']

    # Function to create the environment
    def make_env():
        env = env_class()
        # The MetaworldWrapper is essential for training
        return MetaworldWrapper(env, ml1)

    # Use a vectorized environment for stable-baselines3
    # This allows for easy scaling to multiple parallel environments in the future
    vec_env = DummyVecEnv([make_env])

    # --- 2. Create and Train the PPO Agent ---
    # log_dir = "./ppo_pick_place_tensorboard/"
    # os.makedirs(log_dir, exist_ok=True)

    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        # tensorboard_log=log_dir,
        device="auto",  # Automatically uses GPU if available
    )

    print("Starting training...")
    # Train the agent. The total_timesteps can be increased for better performance.
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    # --- 3. Save the Model and Clean Up ---
    model_path = os.path.join(".", "ppo_pick_place_model2.zip")
    model.save(model_path)
    vec_env.close()

    print("-" * 20)
    print("Training complete.")
    print(f"Model saved to: {model_path}")
    # print(f"To view TensorBoard logs, run: tensorboard --logdir {log_dir}")
    print("-" * 20)

if __name__ == "__main__":
    main()
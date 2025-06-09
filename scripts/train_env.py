import wandb
from wandb.integration.sb3 import WandbCallback
import os
import sys
import metaworld
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.ppo import PPO
from scripts.wandb_callback import WandbLoggingCallback
import gymnasium as gym
# from scripts.custom_utils import CustomPPO #, CustomEvalCallback, CustomDummyVecEnv, CustomMonitor
# from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# === CONFIGURATION ===
TOTAL_TIMESTEPS = 1_000_000
ENV_NAME = 'custom-two-balls'  # Name of the environment to use

# --- Wrapper for Metaworld Environments ---
# This wrapper is essential for training, as it handles setting a new random
# task for the environment at the beginning of each episode.
class MetaworldWrapper(gym.Wrapper):
    """
    Wrapper to set a new random task from a Metaworld benchmark for each episode.
    """
    def __init__(self, env, benchmark):
        super().__init__(env)
        self.benchmark = benchmark
        # Create an iterator for the tasks to cycle through them
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
def make_env_func():
    def _init():
        """
        Creates and wraps the Metaworld environment.
        """
        # Create the benchmark and get the environment class
        ml1 = metaworld.ML1(ENV_NAME)
        env_class = ml1.train_classes[ENV_NAME]

        # Instantiate the environment
        env = env_class()

        # Apply the crucial wrapper
        env = MetaworldWrapper(env, ml1)

        # Monitor is helpful for tracking episode stats
        env = Monitor(env)
        return env
    return _init

def make_env():
    return make_env_func()()

class VerboseEvalCallback(EvalCallback):
# class VerboseEvalCallback(CustomEvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self.last_mean_reward
            print(f"🧪 Evaluation @ step {self.num_timesteps}: mean reward = {mean_reward:.2f}")
            if self.best_mean_reward is not None:
                print(f"⭐ Best mean reward so far: {self.best_mean_reward:.2f}")

        return result

def main():
    # Add the root directory (metaworld_repo/) to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    LOG_DIR = "./ppo_model/two_balls"
    MODEL_DIR = os.path.join(LOG_DIR, "models")
    LOGS_DIR = os.path.join(LOG_DIR, "logs")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    for i in range(5):
        i = i+10
        wandb.init(
            project="llm_rl",
            name=None,
            # name="PPO_Metaworld_300k",
            config={
                "total_timesteps": TOTAL_TIMESTEPS,
                "policy": "MlpPolicy",
                "env": "TwoBalls",
                "algo": "PPO",
            },
            sync_tensorboard=False,
            monitor_gym=True,
            save_code=True,
        )

        env = DummyVecEnv([make_env_func()])
        model = PPO("MlpPolicy", env, verbose=1)
        # env = CustomDummyVecEnv([make_env()])
        # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"))
        # model = CustomPPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"))

        callbacks = CallbackList([
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=MODEL_DIR,
                verbose=2,
            ),
            WandbLoggingCallback(
                env,
                best_model_save_path=MODEL_DIR,
                eval_freq=5000,
                n_eval=15,
                log_freq=1000,
            ),
        ])

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
        model.save(os.path.join(MODEL_DIR, "ppo_final_" + str(i)))

        print("✅ Training complete.")
        wandb.finish()

if __name__ == "__main__":
    main()
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import sys
import metaworld
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from scripts.wandb_callback import WandbLoggingCallback
import gymnasium as gym
# from scripts.custom_utils import CustomPPO #, CustomEvalCallback, CustomDummyVecEnv, CustomMonitor
# from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

# """
# This is what hyperparameters were tested for 70k timesteps on the pick-place-v3 environment.

# DEV TESTING:
# "learning_rate": 0.0001 0.0003 0.0005
# "clip_range": 0.1 0.05
# "ent_coef": 0.01 0.001
# "n_steps": 2048 4096
# """
# === CONFIGURATION ===
TEST_NUMBER = sys.argv[1:][0]
print(f"Running test number: {TEST_NUMBER}")
assert TEST_NUMBER in ['1', '2', '3'], "Test number must be 1, 2, or 3."
TOTAL_TIMESTEPS = 70_000
ENV_NAME = 'reach-v3'
PPO_HYPERPARAMS = {
    1: {
        "n_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "n_epochs": 4000,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "policy_kwargs": dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn='tanh',
            ),
    },
    2: {
        "n_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "n_epochs": 4000,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn='tanh',
            ),
    },
    3: {
        "n_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "n_epochs": 4000,
        "clip_range": 0.1,
        "ent_coef": 0.0,
        "policy_kwargs": dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn='tanh',
            ),
    },
}

class StateRewardPrintCallback(BaseCallback):
    """
    A custom callback to print the reward and state at each step.
    Note: This will print a lot of information and can slow down training.
    """
    def __init__(self, verbose=0):
        super(StateRewardPrintCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals contains the variables from the training loop.
        # For a vectorized environment, 'rewards' and 'new_observations' are arrays.
        # We access the first element since you are using a DummyVecEnv with one environment.
        reward = self.locals['rewards'][0]
        state = self.locals['new_obs'][0]

        # Printing at every step can be overwhelming.
        # Consider printing only periodically, for example, every 1000 steps.
        if self.num_timesteps % 1000 == 0:
            print(f"\n--- Step: {self.num_timesteps} ---")
            print(f"  - Reward: {reward:.5f}")
            # The state vector can be large, so we print its shape and a small slice.
            print(f"  - State (Observation): {state}")
            print("--------------------------")

        return True # Return True to continue training

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
        Creates and wraps the Metaworld environment correctly.
        """
        # Create the benchmark and get the environment class
        mt1 = metaworld.MT1(ENV_NAME)
        env_class = mt1.train_classes[ENV_NAME]

        # Get all available training tasks
        tasks = mt1.train_tasks

        # Instantiate the environment
        env = env_class()

        # Set partially observable to False
        env._partially_observable = False

        # Set a SINGLE task for the entire training run.
        # It's common practice to use the first task for consistency.
        env.set_task(tasks[0])

        # Set wrappers
        env = Monitor(env)
        #     gym.wrappers.RecordEpisodeStatistics(
        #         NormalizeReward(
        #             NormalizeObservation(
        #                 MetaworldWrapper(env, mt1)
        #             )
        #         )
        #     )
        # )
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

    LOG_DIR = "./ppo_models/pick_place"
    MODEL_DIR = os.path.join(LOG_DIR, "models")
    LOGS_DIR = os.path.join(LOG_DIR, "logs")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # for i in range(1, 2):
    model_name = f"Reach{TEST_NUMBER}"
    wandb.init(
        project="metaworld",
        name=model_name,
        # name="PPO_Metaworld_300k",
        config=PPO_HYPERPARAMS[TEST_NUMBER],
        sync_tensorboard=False,
        monitor_gym=True,
        save_code=True,
    )

    env = DummyVecEnv([make_env_func()])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        **(PPO_HYPERPARAMS[TEST_NUMBER]),
    )

    wandb.watch(model.policy, log="all", log_freq=1000)

    callbacks = CallbackList([
        StateRewardPrintCallback(),
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
    model.save(os.path.join(MODEL_DIR, f"reach_{TEST_NUMBER}"))

    print("✅ Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
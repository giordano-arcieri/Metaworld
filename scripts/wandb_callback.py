from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
import numpy as np
from collections import deque

#  info = {
#         "success": grasp_success,
#         "false_success": False,
#         "reach_reward": reach_reward,
#         "grasp_reward": grasp_reward,
#         "total_reward": reward,
#         "tcp_to_obj": tcp_to_obj,
#         "tcp_open": tcp_opened,
#     }

class WandbLoggingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval=5, log_freq=1000, verbose=0, best_model_save_path=None):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.recent_successes = deque(maxlen=100)
        self.eval_callback = EvalCallback(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval,
            deterministic=True,
            render=False,
        )

        self.episode_rewards = []
        self.episode_successes = []
        self.rollout_rewards = []
        self.rollout_successes = []
        self.rollout_ep_lengths = []
        self.reach_rewards = []
        self.reach_bonuses = []
        self.grasp_bonuses = []
        self.penalties = []

        self.best_model_save_path = best_model_save_path

    def _on_training_start(self) -> None:
        self.eval_callback.init_callback(self.model)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if "total_reward" in info:
                self.episode_rewards.append(info["total_reward"])
                self.episode_successes.append(info.get("success", 0.0))
            if "reach_reward" in info:
                self.reach_rewards.append(info.get("reach_reward", 0.0))
            if "grasp_reward" in info:
                self.reach_bonuses.append(info.get("grasp_reward", 0.0))
            if "penalty" in info:
                self.penalties.append(info.get("penalty", 0.0))

            if done:
                ep_reward = np.sum(self.episode_rewards)
                ep_success = np.sum(self.episode_successes)
                ep_success_avg = float(np.any(self.episode_successes))
                ep_len = len(self.episode_rewards)

                wandb.log({
                    "episode/total_reward": ep_reward,
                    "episode/reach_reward": np.sum(self.reach_rewards),
                    "episode/reach_bonus": np.sum(self.reach_bonuses),
                    "episode/penalty": np.sum(self.penalties),
                    "episode/success_rate": ep_success,
                    "episode/length": ep_len,
                }, step=self.num_timesteps)

                self.rollout_rewards.append(ep_reward)
                self.rollout_successes.append(ep_success)
                self.rollout_ep_lengths.append(ep_len)
                self.recent_successes.append(ep_success_avg)

                self.episode_rewards.clear()
                self.episode_successes.clear()
                self.reach_rewards.clear()
                self.penalties.clear()

        # Log rollout metrics once per log_freq
        if self.n_calls % self.log_freq == 0 and self.rollout_rewards:
            wandb.log({
                "rollout/mean_reward": np.mean(self.rollout_rewards),
                "rollout/success_rate": np.mean(self.rollout_successes),
                "rollout/success_rate_moving_avg": np.mean(self.recent_successes),
                "rollout/mean_ep_len": np.mean(self.rollout_ep_lengths)
            }, step=self.num_timesteps)

        # Step the EvalCallback and log eval stats
        self.eval_callback.num_timesteps = self.num_timesteps
        self.eval_callback.n_calls = self.n_calls
        self.eval_callback._on_step()
        if self.eval_callback.n_calls % self.eval_callback.eval_freq == 0:
            # if self.eval_callback.evaluations_successes:
            eval_success_rate = np.mean(self.eval_callback._is_success_buffer)
            wandb.log({
                "eval/mean_reward": self.eval_callback.last_mean_reward,
                "eval/best_mean_reward": self.eval_callback.best_mean_reward,
                "eval/success_rate": eval_success_rate,
            }, step=self.num_timesteps)

            if self.eval_callback.last_mean_reward > self.eval_callback.best_mean_reward and self.best_model_save_path is not None:
                self.eval_callback.best_mean_reward = self.eval_callback.last_mean_reward
                self.model.save(self.best_model_save_path)
                print(f"📦 Saved new best model at step {self.num_timesteps} with reward {self.eval_callback.last_mean_reward:.2f}")
                wandb.save(self.best_model_save_path)

        return True
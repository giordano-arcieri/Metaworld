from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
import numpy as np
from collections import deque

# info: { "success": float(pressed),
#         "false_success": float(false_pressed), 
#         "btn_press_depth": float(press_depth),
#         "btn_false_press_depth": float(false_press_depth), 
#         "unscaled_reward": reward,
#         "reach_rew": reach_reward, 
#         "reach_bon": bonus_reach, 
#         "press_bon": press_bonus, 
#         "penalty": penalty, }

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
        self.press_depths = [] 
        self.false_press_depths = []
        self.reach_rewards = []
        self.reach_bonuses = [] 
        self.press_bonuses = []
        self.penalties = []

    def _on_training_start(self) -> None:
        self.eval_callback.init_callback(self.model)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if "unscaled_reward" in info:
                self.episode_rewards.append(info["unscaled_reward"])
                self.episode_successes.append(info.get("success", 0.0))
            if "btn_press_depth" in info: 
                self.press_depths.append(info.get("btn_press_depth", 0.0))
                self.false_press_depths.append(info.get("btn_false_press_depth", 0.0))
            if "reach_rew" in info: 
                self.reach_rewards.append(info.get("reach_rew", 0.0))
            if "reach_bon" in info: 
                self.reach_bonuses.append(info.get("reach_bon", 0.0))
            if "press_bon" in info: 
                self.press_bonuses.append(info.get("press_bon", 0.0))
            if "penalty" in info: 
                self.penalties.append(info.get("penalty", 0.0))

            if done:
                ep_reward = np.sum(self.episode_rewards)
                ep_success = np.sum(self.episode_successes)
                ep_success_avg = float(np.any(self.episode_successes))
                ep_len = len(self.episode_rewards)

                wandb.log({
                    "episode/total_reward": ep_reward,
                    "episode/reach_rew": np.sum(self.reach_rewards),
                    "episode/press_bonus": np.sum(self.press_bonuses),
                    "episode/penalty": np.sum(self.penalties),
                    "episode/success_rate": ep_success,
                    "episode/press_depth": np.mean(self.press_depths), 
                    "episode/false_press_depth": np.mean(self.false_press_depths), 
                    "episode/length": ep_len,
                }, step=self.num_timesteps)

                self.rollout_rewards.append(ep_reward)
                self.rollout_successes.append(ep_success)
                self.rollout_ep_lengths.append(ep_len)
                self.recent_successes.append(ep_success_avg)

                self.episode_rewards.clear()
                self.episode_successes.clear()
                self.press_depths.clear()
                self.false_press_depths.clear()
                self.reach_rewards.clear()
                self.press_bonuses.clear()
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

            if self.eval_callback.last_mean_reward > self.eval_callback.best_mean_reward and best_model_save_path:
                self.eval_callback.best_mean_reward = self.eval_callback.last_mean_reward
                self.model.save(self.best_model_save_path)
                print(f"📦 Saved new best model at step {self.num_timesteps} with reward {self.eval_callback.last_mean_reward:.2f}")
                wandb.save(self.best_model_save_path)

        return True
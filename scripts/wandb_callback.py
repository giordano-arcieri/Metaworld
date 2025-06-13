from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
import numpy as np
from collections import deque

# info = {
#     "success": success,
#     "near_object": near_object,
#     "grasp_success": grasp_success,
#     "grasp_reward": grasp_reward,
#     "in_place_reward": in_place_reward,
#     "obj_to_target": obj_to_target,
#     "unscaled_reward": reward,
# }

class WandbLoggingCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq=10000,
        n_eval=5,
        log_freq=1000,
        verbose=0,
        best_model_save_path=None
    ):
        super().__init__(verbose)
        self.log_freq = log_freq

        # for smoothing eval success
        self.recent_successes = deque(maxlen=100)

        # set up the built-in EvalCallback
        self.eval_callback = EvalCallback(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval,
            deterministic=True,
            render=False,
        )

        # per-episode buffers
        self.episode_rewards       = []
        self.episode_successes     = []
        self.near_objects          = []
        self.grasp_successes       = []
        self.grasp_rewards         = []
        self.in_place_rewards      = []
        self.obj_to_target_dists   = []

        # rollout (across many episodes) stats
        self.rollout_rewards     = []
        self.rollout_successes   = []
        self.rollout_ep_lengths  = []

        self.best_model_save_path = best_model_save_path

    def _on_training_start(self) -> None:
        self.eval_callback.init_callback(self.model)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            # collect every field if present
            if "unscaled_reward" in info:
                self.episode_rewards.append(info["unscaled_reward"])
                self.episode_successes.append(info.get("success", 0.0))
            if "near_object" in info:
                self.near_objects.append(info.get("near_object", 0.0))
            if "grasp_success" in info:
                self.grasp_successes.append(info.get("grasp_success", 0.0))
            if "grasp_reward" in info:
                self.grasp_rewards.append(info["grasp_reward"])
            if "in_place_reward" in info:
                self.in_place_rewards.append(info["in_place_reward"])
            if "obj_to_target" in info:
                self.obj_to_target_dists.append(info["obj_to_target"])

            if done:
                # compute episode metrics
                ep_reward = np.sum(self.episode_rewards)
                ep_success_rate = float(np.mean(self.episode_successes))
                ep_len = len(self.episode_rewards)

                wandb.log({
                    "episode/total_reward":      ep_reward,
                    "episode/success_rate":      ep_success_rate,
                    "episode/near_object_rate":  np.mean(self.near_objects),
                    "episode/grasp_success_rate":np.mean(self.grasp_successes),
                    "episode/grasp_reward":      np.sum(self.grasp_rewards),
                    "episode/in_place_reward":   np.sum(self.in_place_rewards),
                    "episode/obj_to_target":     np.mean(self.obj_to_target_dists),
                    "episode/length":            ep_len,
                }, step=self.num_timesteps)

                # record for rollout summaries
                self.rollout_rewards.append(ep_reward)
                self.rollout_successes.append(ep_success_rate)
                self.rollout_ep_lengths.append(ep_len)
                self.recent_successes.append(ep_success_rate)

                # clear buffers for next episode
                self.episode_rewards.clear()
                self.episode_successes.clear()
                self.near_objects.clear()
                self.grasp_successes.clear()
                self.grasp_rewards.clear()
                self.in_place_rewards.clear()
                self.obj_to_target_dists.clear()

        # rollout-level logging
        if (self.n_calls % self.log_freq == 0) and self.rollout_rewards:
            wandb.log({
                "rollout/mean_reward":           np.mean(self.rollout_rewards),
                "rollout/mean_success_rate":     np.mean(self.rollout_successes),
                "rollout/success_rate_moving_avg": np.mean(self.recent_successes),
                "rollout/mean_ep_length":        np.mean(self.rollout_ep_lengths),
            }, step=self.num_timesteps)

        # step the EvalCallback
        self.eval_callback.num_timesteps = self.num_timesteps
        self.eval_callback.n_calls      = self.n_calls
        self.eval_callback._on_step()
        # log eval stats when it's evaluation time
        if self.eval_callback.n_calls % self.eval_callback.eval_freq == 0:
            eval_success_rate = np.mean(self.eval_callback._is_success_buffer)
            wandb.log({
                "eval/mean_reward":      self.eval_callback.last_mean_reward,
                "eval/best_mean_reward": self.eval_callback.best_mean_reward,
                "eval/success_rate":     eval_success_rate,
            }, step=self.num_timesteps)

            # save best model
            if (
                self.best_model_save_path is not None
                and self.eval_callback.last_mean_reward > self.eval_callback.best_mean_reward
            ):
                self.eval_callback.best_mean_reward = self.eval_callback.last_mean_reward
                self.model.save(self.best_model_save_path)
                print(
                    f"📦 Saved new best model at step {self.num_timesteps}"
                    f" with reward {self.eval_callback.last_mean_reward:.2f}"
                )
                wandb.save(self.best_model_save_path)

        return True

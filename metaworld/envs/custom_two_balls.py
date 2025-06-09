from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils

COLOR_MAP = {
    'red': np.array([1, 0], dtype=np.float32),
    'green': np.array([0, 1], dtype=np.float32),
}

BUTTON_VARIANTS = ['red', 'green']


class CustomTwoBalls(SawyerXYZEnv):
    """CustomTwoBalls.

    This environment is a variant of SawyerPickPlace, but the goal is
    just to pick a single object instead of pick and place. Also there
    are two objects in the environment, but only one of them is
    considered the main object to be picked. The state of the environment
    includes the color of the object, which is used to determine
    which object is the main one to be picked.
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        # Set the color of the target button randomly
        self.target_variant = 'red'  # random.choice(BUTTON_VARIANTS)
        self.color_vec = COLOR_MAP[self.target_variant]

        # Bounds for the robot hand workspace.
        # Actions outside this box are clipped by the environment.
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)

        # Bounds for initial object placement.
        # Both balls will be reset randomly within this region.
        red_ball_low = (-0.4, 0.5, 0.02)
        red_ball_high = (0.4, 0.8, 0.02)

        # Bounds for initial object placement.
        # Both balls will be reset randomly within this region.
        green_ball_low = (-0.4, 0.5, 0.02)
        green_ball_high = (0.4, 0.8, 0.02)

        # Initialize the base SawyerXYZEnv: 
        # - hand_low/hand_high define the reachable workspace
        # - render_mode, camera_name/id configure the viewer if needed
        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        # Default starting configuration for the first ball and the robot hand
        self.init_config: InitConfigDict = {
            "red_ball_init_angle": 0.3,
            "red_ball_init_pos": np.array([-0.6, 0.6, 0.02]),
            "green_ball_init_angle": 0.3,
            "green_ball_init_pos": np.array([0.6, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }


        # Unpack and store the init_config values into attributes for reset_model()
        self.red_ball_init_angle = self.init_config["red_ball_init_angle"]
        self.red_ball_init_pos = self.init_config["red_ball_init_pos"]
        self.green_ball_init_angle = self.init_config["green_ball_init_angle"]
        self.green_ball_init_pos = self.init_config["green_ball_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        # Define a 6‑D Box space: [obj1_xyz, obj2_xyz]
        self._random_reset_space = Box(
            np.hstack((red_ball_low, green_ball_low)),
            np.hstack((red_ball_high, green_ball_high)),
            dtype=np.float64,
        )

        # Counter tracking how many times reset_model() has been called
        self.num_resets = 0

        # Clear obj_init_pos so that the first reset uses a random position
        self.red_ball_init_pos = None
        self.green_ball_init_pos = None

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/custom_two_balls.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        assert self.red_ball_init_pos is not None and self.green_ball_init_pos is not None

        grasp_success = float(
            self.touching_main_object
            and (tcp_open > 0)
            and (obj[2] - 0.02 > self.red_ball_init_pos[2])
        )

        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return self.data.geom(f"{self.target_variant}Geom").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return np.hstack((
            self.get_body_com(f"ball_red"),
            self.get_body_com(f"ball_green"),
        ))

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.hstack((
            Rotation.from_matrix(
                self.data.geom("redGeom").xmat.reshape(3, 3)
            ).as_quat(),
            Rotation.from_matrix(
                self.data.geom("greenGeom").xmat.reshape(3, 3)
            ).as_quat(),
        ))

    def _get_color_objects(self) -> npt.NDArray[Any]:
        # Returns the color of the balls (red and green)
        return np.hstack((
            COLOR_MAP["red"],
            COLOR_MAP["green"],
        ))

    def fix_extreme_red_ball_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("ball_red")[:2] - self.get_body_com("ball_red")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return np.array(
            [adjusted_pos[0], adjusted_pos[1], self.get_body_com("ball_red")[-1]]
        )

    def fix_extreme_green_ball_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        diff = self.get_body_com("ball_green")[:2] - self.get_body_com("ball_green")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        return np.array(
            [adjusted_pos[0], adjusted_pos[1], self.get_body_com("ball_green")[-1]]
        )

    def reset_model(self) -> npt.NDArray[np.float64]:
        # 1) Randomize ball positions within obj_low/obj_high
        # 2) Reset the robot hand to its home position
        self._reset_hand()

        # Set the color of the target button randomly
        self.target_variant = 'red'  # random.choice(BUTTON_VARIANTS)
        self.color_vec = COLOR_MAP[self.target_variant]

        # Fix the extreme positions of the balls
        self.red_ball_init_pos = self.fix_extreme_red_ball_pos(self.init_config["red_ball_init_pos"])
        self.red_ball_init_angle = self.init_config["red_ball_init_angle"]
        self.green_ball_init_pos = self.fix_extreme_green_ball_pos(self.init_config["green_ball_init_pos"])
        self.green_ball_init_angle = self.init_config["green_ball_init_angle"]

        # Get random state vector for the environment
        rand_vec = self._get_state_rand_vec()
        # print("Random state vector:", rand_vec)

        # Set the random positions
        self._target_pos  = rand_vec[0:3]
        self.red_ball_init_pos = rand_vec[0:3]
        self.green_ball_init_pos = rand_vec[3:6]

        # Initialize TCP and pad positions
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self._set_obj_xyz(np.concatenate((self.red_ball_init_pos, self.green_ball_init_pos)))

        ### FIX LATER
        self.objHeight = self.data.geom("redGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04

        self.maxPlacingDist = (
            np.linalg.norm(
                np.array(
                    [self.red_ball_init_pos[0], self.red_ball_init_pos[1], self.heightTarget]
                )
                - np.array(self._target_pos)
            )
            + self.heightTarget
        )

        self.maxPushDist = np.linalg.norm(
            self.red_ball_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        return self._get_obs()

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float = 0,  # All of these args are unused, just here to match
        pad_success_thresh: float = 0,  # the parent's type signature
        object_reach_radius: float = 0,
        xz_thresh: float = 0,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_pos[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_pos[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        y_caging = reward_utils.hamacher_product(left_caging, right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_pos) + np.array([0.0, -obj_pos[1], 0.0])
        tcp_obj_norm_x_z = float(np.linalg.norm(tcp_xz - obj_position_x_z, ord=2))

        # used for computing the tcp to object object margin in the x_z plane
        assert self.red_ball_init_pos is not None
        init_obj_x_z = self.red_ball_init_pos + np.array([0.0, -self.red_ball_init_pos[1], 0.0])
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])
        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )

        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        gripper_closed = min(max(0, action[-1]), 1)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        caging_and_gripping = (caging_and_gripping + caging) / 2
        return caging_and_gripping

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.red_ball_init_pos is not None
        if True:
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened = obs[3]

            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            # The `reach_margin` should be the typical distance from the gripper
            # to the object at the start of an episode.
            reach_margin = np.linalg.norm(self.hand_init_pos - self.red_ball_init_pos)
            reach_reward = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.02), # Target distance is 0, with a tolerance of 0.02
                margin=reach_margin,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(action, obj)

            reward = reach_reward + 5 * object_grasped

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                -1,
                object_grasped,
                -1,
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos
            del obs

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            # assert np.all(goal == self._get_site_pos("goal"))

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50
            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance):
                self.pickCompleted = True
            else:
                self.pickCompleted = False

            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

            hScale = 100
            if self.pickCompleted and not (objDropped):
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                pickRew = hScale * min(heightTarget, objPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )

            cond = self.pickCompleted and (reachDist < 0.1) and not (objDropped)
            if cond:
                placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                    np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3)
                )
                placeRew = max(placeRew, 0)
            else:
                placeRew = 0

            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew

            return float(reward), 0.0, 0.0, float(placingDist), 0.0, 0.0

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the object.

        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        if pos.shape == (6,):
            # two objects: first at qpos[9:12], second at qpos[12:15]
            qpos[9:12]   = pos[:3].copy()   # red ball
            qpos[16:19]  = pos[3:6].copy()   # green ball
            # zero both bodies' vel (3 lin + 3 ang each)
            qvel[9:] = 0
        else:
            raise ValueError(f"_set_obj_xyz() expects pos.shape (6,), got {pos.shape}")

        self.set_state(qpos, qvel)
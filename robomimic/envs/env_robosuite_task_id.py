
import json
import numpy as np
from copy import deepcopy

import mujoco_py
import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
import robomimic.envs.env_robosuite as ER


class EnvRobosuiteWithTask(ER.EnvRobosuite):
    def __init__(self, env, task_id):
        self.env = env
        self.task_id = task_id

    # def step(self, action):
    #     """
    #     Step in the environment with an action.
    #
    #     Args:
    #         action (np.array): action to take
    #
    #     Returns:
    #         observation (dict): new observation dictionary
    #         reward (float): reward for this step
    #         done (bool): whether the task is done
    #         info (dict): extra information
    #     """
    #     obs, r, done, info = self.env.step(action)
    #     obs = self.get_observation(obs)
    #     return obs, r, self.is_done(), info
    #
    # def reset(self):
    #     """
    #     Reset environment.
    #
    #     Returns:
    #         observation (dict): initial observation dictionary.
    #     """
    #     di = self.env.reset()
    #     return self.get_observation(di)
    #
    # def reset_to(self, state):
    #     """
    #     Reset to a specific simulator state.
    #
    #     Args:
    #         state (dict): current simulator state that contains one or more of:
    #             - states (np.ndarray): initial state of the mujoco environment
    #             - model (str): mujoco scene xml
    #
    #     Returns:
    #         observation (dict): observation dictionary after setting the simulator state (only
    #             if "states" is in @state)
    #     """
    #     should_ret = False
    #     if "model" in state:
    #         self.reset()
    #         xml = postprocess_model_xml(state["model"])
    #         self.env.reset_from_xml_string(xml)
    #         self.env.sim.reset()
    #         if not self._is_v1:
    #             # hide teleop visualization after restoring from model
    #             self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
    #             self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    #     if "states" in state:
    #         self.env.sim.set_state_from_flattened(state["states"])
    #         self.env.sim.forward()
    #         should_ret = True
    #
    #     if "goal" in state:
    #         self.set_goal(**state["goal"])
    #     if should_ret:
    #         # only return obs if we've done a forward call - otherwise the observations will be garbage
    #         return self.get_observation()
    #     return None
    #
    # def render(self, mode="human", height=None, width=None, camera_name="agentview"):
    #     """
    #     Render from simulation to either an on-screen window or off-screen to RGB array.
    #
    #     Args:
    #         mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
    #         height (int): height of image to render - only used if mode is "rgb_array"
    #         width (int): width of image to render - only used if mode is "rgb_array"
    #         camera_name (str): camera name to use for rendering
    #     """
    #     if mode == "human":
    #         cam_id = self.env.sim.model.camera_name2id(camera_name)
    #         self.env.viewer.set_camera(cam_id)
    #         return self.env.render()
    #     elif mode == "rgb_array":
    #         return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
    #     else:
    #         raise NotImplementedError("mode={} is not implemented".format(mode))
    #
    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide
                as a dictionary. If not provided, will be queried from robosuite.
        """
        obs = super(EnvRobosuiteWithTask, self).get_observation(di)
        print(obs)
        return obs
    #
    # def get_state(self):
    #     """
    #     Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
    #     """
    #     xml = self.env.sim.model.get_xml() # model xml file
    #     state = np.array(self.env.sim.get_state().flatten()) # simulator state
    #     return dict(model=xml, states=state)
    #
    # def get_reward(self):
    #     """
    #     Get current reward.
    #     """
    #     return self.env.reward()
    #
    # def get_goal(self):
    #     """
    #     Get goal observation. Not all environments support this.
    #     """
    #     return self.get_observation(self.env._get_goal())
    #
    # def set_goal(self, **kwargs):
    #     """
    #     Set goal observation with external specification. Not all environments support this.
    #     """
    #     return self.env.set_goal(**kwargs)
    #
    # def is_done(self):
    #     """
    #     Check if the task is done (not necessarily successful).
    #     """
    #
    #     # Robosuite envs always rollout to fixed horizon.
    #     return False
    #
    # def is_success(self):
    #     """
    #     Check if the task condition(s) is reached. Should return a dictionary
    #     { str: bool } with at least a "task" key for the overall task success,
    #     and additional optional keys corresponding to other task criteria.
    #     """
    #     succ = self.env._check_success()
    #     if isinstance(succ, dict):
    #         assert "task" in succ
    #         return succ
    #     return { "task" : succ }
    #
    # @property
    # def action_dimension(self):
    #     """
    #     Returns dimension of actions (int).
    #     """
    #     return self.env.action_spec[0].shape[0]
    #
    # @property
    # def name(self):
    #     """
    #     Returns name of environment name (str).
    #     """
    #     return self._env_name
    #
    # @property
    # def type(self):
    #     """
    #     Returns environment type (int) for this kind of environment.
    #     This helps identify this env class.
    #     """
    #     return EB.EnvType.ROBOSUITE_TYPE
    #
    # def serialize(self):
    #     """
    #     Save all information needed to re-instantiate this environment in a dictionary.
    #     This is the same as @env_meta - environment metadata stored in hdf5 datasets,
    #     and used in utils/env_utils.py.
    #     """
    #     return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))
    #
    # @classmethod
    # def create_for_data_processing(
    #     cls,
    #     env_name,
    #     camera_names,
    #     camera_height,
    #     camera_width,
    #     reward_shaping,
    #     **kwargs,
    # ):
    #     """
    #     Create environment for processing datasets, which includes extracting
    #     observations, labeling dense / sparse rewards, and annotating dones in
    #     transitions.
    #
    #     Args:
    #         env_name (str): name of environment
    #         camera_names (list of str): list of camera names that correspond to image observations
    #         camera_height (int): camera height for all cameras
    #         camera_width (int): camera width for all cameras
    #         reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
    #     """
    #     is_v1 = (robosuite.__version__.split(".")[0] == "1")
    #     has_camera = (len(camera_names) > 0)
    #
    #     new_kwargs = {
    #         "reward_shaping": reward_shaping,
    #     }
    #
    #     if has_camera:
    #         if is_v1:
    #             new_kwargs["camera_names"] = list(camera_names)
    #             new_kwargs["camera_heights"] = camera_height
    #             new_kwargs["camera_widths"] = camera_width
    #         else:
    #             assert len(camera_names) == 1
    #             if has_camera:
    #                 new_kwargs["camera_name"] = camera_names[0]
    #                 new_kwargs["camera_height"] = camera_height
    #                 new_kwargs["camera_width"] = camera_width
    #
    #     kwargs.update(new_kwargs)
    #
    #     # also initialize obs utils so it knows which modalities are image modalities
    #     image_modalities = list(camera_names)
    #     if is_v1:
    #         image_modalities = ["{}_image".format(cn) for cn in camera_names]
    #     elif has_camera:
    #         # v0.3 only had support for one image, and it was named "image"
    #         assert len(image_modalities) == 1
    #         image_modalities = ["image"]
    #     obs_modality_specs = {
    #         "obs": {
    #             "low_dim": [], # technically unused, so we don't have to specify all of them
    #             "image": image_modalities,
    #         }
    #     }
    #     ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
    #
    #     # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
    #     return cls(
    #         env_name=env_name,
    #         render=False,
    #         render_offscreen=has_camera,
    #         use_image_obs=has_camera,
    #         postprocess_visual_obs=False,
    #         **kwargs,
    #     )
    #
    # @property
    # def rollout_exceptions(self):
    #     """
    #     Return tuple of exceptions to except when doing rollouts. This is useful to ensure
    #     that the entire training run doesn't crash because of a bad policy that causes unstable
    #     simulation computations.
    #     """
    #     return (mujoco_py.builder.MujocoException)
    #
    # def __repr__(self):
    #     """
    #     Pretty-print env description.
    #     """
    #     return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

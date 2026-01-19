import gymnasium as gym
import numpy as np
import metaworld
import mujoco
import mujoco.viewer

class MetaWorldMT1Wrapper:
    """
    Wraps a Metaworld MT1 environment into a simple interface:
    - reset() -> (image, state, info)
    - step(action) -> (image, state, reward, done, info)
    
    When render_mode='human':
    - Uses MuJoCo's native viewer for display
    - Returns RGB arrays from main environment
    """
    def __init__(self, env_name='push-v3', seed=42, render_mode='rgb_array', camera_name='topview'):
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.seed = seed

        # Main environment
        self.env = gym.make(
            'Meta-World/MT1',
            env_name=env_name,
            seed=seed,
            render_mode="rgb_array",
            camera_name=camera_name
        )

        # MuJoCo viewer
        self.viewer = None
        if render_mode == "human":
            # Access the underlying MuJoCo environment
            self.viewer = mujoco.viewer.launch_passive(
                self.env.unwrapped.model,
                self.env.unwrapped.data
            )

        # Initialization
        obs, _ = self.env.reset(seed=seed)
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

    def _extract_state(self, obs):
        """Extract state from observation."""
        if isinstance(obs, dict):
            if "observation" in obs:
                state = obs["observation"]
            elif "robot_state" in obs or "object_state" in obs:
                state_parts = []
                if "robot_state" in obs:
                    state_parts.append(obs["robot_state"])
                if "object_state" in obs:
                    state_parts.append(obs["object_state"])
                state = np.concatenate(state_parts, axis=-1)
            else:
                raise KeyError(
                    f"No suitable state keys in observation dict. "
                    f"Available keys: {list(obs.keys())}"
                )
        else:
            state = obs
        return np.asarray(state, dtype=np.float32)

    def _get_image(self):
        """Render and return RGB array from main environment."""
        img = self.env.render()
        if img is None:
            raise RuntimeError("render() returned None from rgb_array environment")
        return img.astype(np.uint8)

    def _update_viewer(self):
        """Update MuJoCo viewer if it exists."""
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def reset(self, seed=None):
        """Reset environment."""
        obs, info = self.env.reset(seed=seed)
        state = self._extract_state(obs)
        image = self._get_image()
        
        # Update viewer
        self._update_viewer()
        
        return image, state, info

    def step(self, action):
        """Take step in environment."""
        obs, reward, truncate, terminate, info = self.env.step(action)
        done = truncate or terminate
        state = self._extract_state(obs)
        image = self._get_image()
        
        # Update viewer
        self._update_viewer()
        
        return image, state, reward, done, info

    def close(self):
        """Close environment and viewer."""
        if self.viewer is not None:
            self.viewer.close()
        self.env.close()
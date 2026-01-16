"""Visualize Meta-World expert policy using MuJoCo viewer"""

import argparse
import time
import gymnasium as gym
import metaworld
from metaworld.policies import ENV_POLICY_MAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="push-right-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create environment
    env = gym.make(
        "Meta-World/MT1",
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
    )
    
    # Get expert policy
    obs, info = env.reset(seed=args.seed)
    policy = ENV_POLICY_MAP[args.env_name]()
    
    # Calculate sleep time for target FPS
    sleep_time = 1.0 / args.fps
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        # Render initial state
        env.render()
        time.sleep(0.5) 
        
        while not done and steps < args.max_steps:
            # Get action from expert policy
            action = policy.get_action(obs)
            
            # Step environment
            obs, reward, truncate, terminate, info = env.step(action)
            
            # Render to update viewer
            env.render()
            
            # Sleep to control playback speed
            time.sleep(sleep_time)
            
            total_reward += reward
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1
        
        success = int(info.get('success', 0))
        print(f"Episode {ep+1}: {steps} steps, reward={total_reward:.2f}, success={'o' if success else 'x'}")
        
        # Pause between episodes
        time.sleep(1.0)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
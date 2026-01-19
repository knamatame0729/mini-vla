"""Collect demonstration data from Meta-World MT1 environments using expert policies"""

import os
import argparse
import time
import numpy as np
import gymnasium as gym
import metaworld
from PIL import Image
from metaworld.policies import ENV_POLICY_MAP
from utils.tokenizer import SimpleTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-names", type=str, nargs="+", default=["push-v3", "push-right-v3"])
    parser.add_argument("--camera-name", type=str, default="topview",
                        help="Meta-World camera: corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output-path", type=str, default="data/metaworld_bc.npz")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional sleep between steps for visualization (seconds)")
    parser.add_argument("--instructions", type=str, nargs="+", default=["push the object to the goal", "push the object to the right"],
                        help="Fixed instructions for all episodes")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Resize images to this size (default: 64x64)")
    return parser.parse_args()


def extract_state(obs):
    """
    Meta-World MT1 observations are already flat numpy arrays.
    """
    return np.asarray(obs, dtype=np.float32).ravel()

def collect_task_data(env_name, instruction, args):
    """Collect single demonstration data from a Meta-World MT1 environment using expert policy."""

    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=args.seed,
        render_mode="rgb_array", # gives images
        camera_name=args.camera_name,
    )

    obs, info = env.reset(seed=args.seed)
    policy = ENV_POLICY_MAP[env_name]()

    images = []
    states = []
    actions = []
    texts = []
    task_ids = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < args.max_steps:
            # expert policy action on raw obs
            action = policy.get_action(obs)  # shape (action_dim,)

            # log current transition
            img = env.render() # (H, W, 3) uint8
            # Resize image to reduce memory usage
            img = Image.fromarray(img)
            img = img.resize((args.img_size, args.img_size), Image.BILINEAR)
            img = np.array(img, dtype=np.uint8)
            state = extract_state(obs) # (state_dim,)

            images.append(img)
            states.append(state.copy())
            actions.append(np.asarray(action, dtype=np.float32).copy())
            texts.append(instruction)
            task_ids.append(env_name)

            # step env
            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {ep+1}/{args.episodes} finished after {steps} steps, success={int(info.get('success', 0))}")

    env.close()

    return images, states, actions, texts, task_ids

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Collect data from all tasks
    all_images = []
    all_states = []
    all_actions = []
    all_texts = []
    all_task_ids = []
    all_task_ids = []

    for env_name, instruction in zip(args.env_names, args.instructions):

        images, states, actions, texts, task_ids = collect_task_data(env_name, instruction, args)

        all_images.extend(images)
        all_states.extend(states)
        all_actions.extend(actions)
        all_texts.extend(texts)
        all_task_ids.extend(task_ids)

    # stack arrays
    images = np.stack(all_images, axis=0)   # (N, H, W, 3)
    states = np.stack(all_states, axis=0)   # (N, state_dim)
    actions = np.stack(all_actions, axis=0) # (N, action_dim)

    # tokenize instructions
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(all_texts)
    text_ids_list = [tokenizer.encode(t) for t in all_texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(all_texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=tokenizer.vocab,
        task_ids=np.array(all_task_ids),
    )

    data = np.load('data/metaworld_push_and_right_bc.npz')
    print("Images:", data['images'].shape)
    print("States:", data['states'].shape) 
    print("Actions:", data['actions'].shape)
    print("Text IDs:", data['text_ids'].shape)
    print("Vocab size:", len(data['vocab'].item()))

    print("Saved Meta-World push dataset to", args.output_path)
    print("  images:", images.shape)
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  text_ids:", text_ids.shape)


if __name__ == "__main__":
    main()

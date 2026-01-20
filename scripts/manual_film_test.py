"""Test VLA Diffusion Policy on Meta-World MT1 with Manual FiLM Parameters"""

import os
import argparse
import time
import numpy as np
import torch
import imageio.v2 as imageio
import wandb

from envs.metaworld_env import MetaWorldMT1Wrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on Meta-World MT1")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="Path to trained VLA diffusion checkpoint",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="push-v3",
        help="Meta-World MT1 task name, e.g. push-v3, reach-v3, pick-place-v3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="push the object to the goal",
        help="Language instruction passed to the VLA",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="If set, save each episode as an MP4 video",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Directory to save videos (if --save-video is set)",
    )

    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    d_model = ckpt["d_model"]
    diffusion_T = ckpt["diffusion_T"]

    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_T,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(vocab=vocab)

    return model, tokenizer


def get_interactive_film_params(action_dim, episode_num):
    """
    Prompt user to enter gamma and beta values interactively.
    
    Args:
        action_dim: Dimension of the action space
        episode_num: Current episode number
    
    Returns:
        gamma: List of gamma values
        beta: List of beta values
    """
    print("\n" + "="*60)
    print(f"Episode {episode_num+1}: Enter FiLM parameters")
    print("="*60)
    
    while True:
        try:
            gamma_input = input(f"Enter gamma values ({action_dim} space-separated floats): ").strip()
            gamma = [float(x) for x in gamma_input.split()]
            
            beta_input = input(f"Enter beta values ({action_dim} space-separated floats): ").strip()
            beta = [float(x) for x in beta_input.split()]

            return gamma, beta
                
        except ValueError as e:
            print(f"Error parsing input: {e}. Please enter valid floating point numbers.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Using default values (identity transformation).")
            return [1.0] * action_dim, [0.0] * action_dim


def run_episode_with_diffusion(model, env, text_ids, device, max_steps, episode_num, global_step, save_video=False, log_to_wandb=True):
    """
    Run a single episode using the full diffusion model.
    """
    img, state, info = env.reset()
    step = 0
    ep_reward = 0.0
    frames = [img.copy()]
    last_action = None
    prev_state = state.copy()

    done = False
    while not done and step < max_steps:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0 # (1, 3, H, W)
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        # Move to device
        img_t = img_t.to(device)
        state_t = state_t.to(device)

        # Inference action with diffusion
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            action_t = model.act(img_t, text_ids, state_t)
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.4f} seconds")
        
        last_action = action_t.clone()
        action_np = action_t.squeeze(0).cpu().numpy()

        img, state, reward, done, info = env.step(action_np)
        ep_reward += reward
        step += 1
        frames.append(img.copy())

        # Calculate deltas
        delta_x = state[0] - prev_state[0]
        delta_y = state[1] - prev_state[1]
        delta_z = state[2] - prev_state[2]

        # Log to W&B
        if log_to_wandb:
            wandb.log({
                "global_step": global_step,
                "episode": episode_num,
                "phase": "Diffusion",

                # End effector position
                "position/x": state[0],
                "position/y": state[1],
                "position/z": state[2],
                
                # Position deltas (velocity)
                "delta/x": delta_x,
                "delta/y": delta_y,
                "delta/z": delta_z,

                # Action commands from diffusion
                "action/x": action_np[0],
                "action/y": action_np[1],
                "action/z": action_np[2],
            })

        prev_state = state.copy()
        step += 1
        frames.append(img.copy())

    # Return final image and state for FiLM episode to continue from
    return ep_reward, step, frames, last_action, img, state, global_step


def run_episode_with_manual_film(env, previous_action, gamma, beta, device, max_steps, img_init, state_init, episode_num, global_step, log_to_wandb=True):
    """
    Run an episode using manually specified gamma and beta values to modulate actions.
    Continue from the current state WITHOUT resetting the environment.
    
    Args:
        env: Environment instance
        previous_action: The action to modulate (torch.Tensor)
        gamma: Scale factors (torch.Tensor or list/array)
        beta: Shift factors (torch.Tensor or list/array)
        device: torch device
        max_steps: Maximum steps
        img_init: Current image state from diffusion episode
        state_init: Current robot state from diffusion episode
    """
    # DO NOT reset - continue from current state
    img, state = img_init, state_init
    step = 0
    ep_reward = 0.0
    frames = [img.copy()]


    # Convert gamma and beta to tensors if needed
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=device)

    print(f"gamma: {gamma.shape}, beta: {beta.shape}, previous_action: {previous_action.shape}")
    
    # Ensure correct shape (1, action_dim)
    if gamma.dim() == 1:
        gamma = gamma.unsqueeze(0)
    if beta.dim() == 1:
        beta = beta.unsqueeze(0)
    
    # Apply FiLM transformation: new_action = gamma * previous_action + beta
    new_action = gamma * previous_action + beta
    

    done = False
    current_action = new_action
    prev_state = state.copy()
    pos = np.zeros(3)
    

    # Print action space bounds for Meta-World
    action_low = env.env.action_space.low # (e.g. array([-1., -1., -1., -1.], dtype=float32))
    action_high = env.env.action_space.high # (e.g. array([1., 1., 1., 1.], dtype=float32))
    print(f"Action space bounds: low={action_low}, high={action_high}")

    # Extract gamma and beta as numpy for logging
    gamma_np = gamma.squeeze(0).cpu().numpy()
    beta_np = beta.squeeze(0).cpu().numpy()
    
    while step < max_steps:
        action_np = current_action.squeeze(0).cpu().numpy()

        print(f"Step {step}: action_np = {action_np}")
        print(f"Action dx, dy , dz: {state[:3]}")

        img, state, reward, done, info = env.step(action_np)
        ep_reward += reward
        
        # Calculate errors between desired action and actual action
        error_delta_x = state[0] - prev_state[0]
        error_delta_y = state[1] - prev_state[1]
        error_delta_z = state[2] - prev_state[2]

        # Calculate positions
        pos += np.array([error_delta_x, error_delta_y, error_delta_z])


        # Log to W&B
        if log_to_wandb:
            wandb.log({
                "global_step": global_step,
                "episode": episode_num,
                "phase": "Manual_FiLM",

                # End effector delta position (velocity)
                "action/x": state[0],
                "action/y": state[1],
                "action/z": state[2],

                # Action errors (velocity)
                "action error/x": error_delta_x,
                "action error/y": error_delta_y,
                "action error/z": error_delta_z,

                # Position
                "position/x": pos[0],
                "position/y": pos[1],
                "position/z": pos[2],

                # Action commands after FiLM
                "action/x": action_np[0],
                "action/y": action_np[1],
                "action/z": action_np[2],

                # FiLM parameters (gamma)
                "film/gamma_x": gamma_np[0],
                "film/gamma_y": gamma_np[1],
                "film/gamma_z": gamma_np[2],
                "film/gamma_gripper": gamma_np[3],

                # FiLM parameters (beta)
                "film/error_x": beta_np[0],
                "film/error_y": beta_np[1],
                "film/error_z": beta_np[2],
                "film/error_gripper": beta_np[3],
            })

        prev_state = state.copy()
        step += 1
        global_step += 1
        frames.append(img.copy())

    return ep_reward, step, frames, gamma, beta, state, current_action, global_step


def run_diffusion_then_manual_film(args, model, env, diff_text_ids, device, episode_num, global_step):
    """
    Run one diffusion episode, then prompt for FiLM parameters and run FiLM episode.
    """
    
    # =================== Diffusion Episode ===================
    ep_reward, step, frames, last_action, final_img, final_state, global_step = run_episode_with_diffusion(
        model, env, diff_text_ids, device, args.max_steps, episode_num, global_step, args.save_video, log_to_wandb=False
    )
    
    print("\n")
    print(f"[Episode {episode_num+1} (VLA)]: reward={ep_reward:.3f}, steps={step}")

    # Save and Log diffusion episode video
    if args.save_video:
        video_path = os.path.join(args.video_dir, f"{args.env_name}_episode{episode_num}_diffusion.mp4")
        with imageio.get_writer(video_path, fps=30) as writer:
            for f in frames:
                f_rot = np.rot90(f, 2)
                writer.append_data(f_rot)

        wandb.log({f"eval/video_diffusion_episode{episode_num+1}": wandb.Video(video_path, format="mp4")})

    # Get manual FiLM parameters AFTER diffusion episode completes
    manual_gamma, manual_beta = get_interactive_film_params(env.action_dim, episode_num)
    
    # Run FiLM episode continuing from current state (no reset)
    ep_reward, step, frames, gamma, beta, final_state_film, new_action, global_step = run_episode_with_manual_film(
        env,
        last_action, 
        manual_gamma,
        manual_beta,
        device, 
        args.max_steps,
        final_img,
        final_state,
        episode_num,
        global_step,
        log_to_wandb=True
    )

    print(f"\n======= Summary ========")
    print(f"  gamma: {gamma.squeeze().cpu().numpy()}")
    print(f"  beta: {beta.squeeze().cpu().numpy()}")
    print(f"  Previous action from diffusion: {last_action.squeeze().cpu().numpy()}")
    print(f"  Last end effector position (XYZ): {final_state[:3]}")
    print(f"  New action (gamma * prev + beta): {new_action.squeeze().cpu().numpy()}")
    print(f"  End effector position after FiLM (XYZ): {final_state_film[:3]}")
    print("="*24 + "\n")
    

    # Save and Log FiLM episode video
    if args.save_video:
        video_path = os.path.join(args.video_dir, f"{args.env_name}_episode{episode_num}_manual_film.mp4")
        with imageio.get_writer(video_path, fps=30) as writer:
            for f in frames:
                f_rot = np.rot90(f, 2)
                writer.append_data(f_rot)

        wandb.log({f"eval/video_manual_film_episode{episode_num+1}": wandb.Video(video_path, format="mp4")})
    
    input("Press Enter to continue to the next episode...")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize W&B for evaluation
    wandb.init(
        project="mini-vla",
        job_type="evaluation",
        config={
            "checkpoint": args.checkpoint,
            "env_name": args.env_name,
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "instruction": args.instruction,
        },
        tags=["evaluation", "manual_film", args.env_name],
    )

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # encode instruction
    diff_tokens = tokenizer.encode(args.instruction)
    diff_text_ids = torch.tensor(diff_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # environment
    env = MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="human",
        camera_name="corner2",
    )
    
    print(f"[test] Meta-World MT1 env: {args.env_name}")
    print(f"[test] state_dim={env.state_dim}, action_dim={env.action_dim}, obs_shape={env.obs_shape}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    global_step = 0

    # Run evaluation episodes until user iterrupts
    try:
        for ep in range(args.episodes):
            run_diffusion_then_manual_film(args, model, env, diff_text_ids, device, ep, global_step)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print("[test] Exiting...")
        
    env.close()
    wandb.finish()
    print("[test] Done.")


if __name__ == "__main__":
    main()
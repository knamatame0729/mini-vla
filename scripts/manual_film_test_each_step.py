"""Test VLA Diffusion Policy on Meta-World MT1 with Manual FiLM Parameters"""

import os
import argparse
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
        default=150,
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


def run_episode_with_diffusion(model, env, text_ids, device, max_steps, save_video=False):
    """
    Run a single episode using the full diffusion model.
    """
    img, state, info = env.reset()
    step = 0
    ep_reward = 0.0
    frames = [img.copy()]
    last_action = None

    done = False
    while not done and step < max_steps:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0 # (1, 3, H, W)
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        # Move to device
        img_t = img_t.to(device)
        state_t = state_t.to(device)

        # Inference action with diffusion
        with torch.no_grad():
            action_t = model.act(img_t, text_ids, state_t)

        print(f" Diffusion step {step}: action_t = {action_t.squeeze(0).cpu().numpy()}")
        
        last_action = action_t.clone()
        action_np = action_t.squeeze(0).cpu().numpy()

        img, state, reward, done, info = env.step(action_np)
        ep_reward += reward
        step += 1
        frames.append(img.copy())


    # Return final image and state for FiLM episode to continue from
    return ep_reward, step, frames, last_action, img, state


def run_single_step_with_manual_film(env, previous_action, gamma, beta, device, max_steps, img_init, state_init):
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

    # Convert gamma and beta to tensors if needed
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
    
    # Ensure correct shape (1, action_dim)
    if gamma.dim() == 1:
        gamma = gamma.unsqueeze(0)
    if beta.dim() == 1:
        beta = beta.unsqueeze(0)
    
    # Apply FiLM transformation: new_action = gamma * previous_action + beta
    new_action = gamma * previous_action + beta

    action_np = new_action.squeeze(0).cpu().numpy()

    print(f" New action after FiLM: {action_np}")

    # Step once with the new action
    img, state, reward, done, info = env.step(action_np)

    return reward, done, img.copy(), new_action, img, state


def run_diffusion_then_manual_film(args, model, env, diff_text_ids, device, episode_num):
    """
    Run one diffusion episode, then prompt for FiLM parameters and run FiLM episode.
    """

    # =================== Diffusion Episode ===================
    ep_reward, step, frames, last_action, final_img, final_state = run_episode_with_diffusion(
        model, env, diff_text_ids, device, args.max_steps, args.save_video
    )
    
    print("\n")
    print(f"[Episode {episode_num+1} (VLA)]: reward={ep_reward:.3f}, steps={step}")
    print(f" Current position (XYZ): {final_state[:3]}")

    # Save and Log diffusion episode video
    if args.save_video:
        video_path = os.path.join(args.video_dir, f"{args.env_name}_episode{episode_num}_diffusion.mp4")
        with imageio.get_writer(video_path, fps=30) as writer:
            for f in frames:
                f_rot = np.rot90(f, 2)
                writer.append_data(f_rot)

        wandb.log({f"eval/video_diffusion_episode{episode_num+1}": wandb.Video(video_path, format="mp4")})

    # =================== Manual FiLM Episode ===================
    curretnt_action = last_action
    current_img = final_img
    current_state = final_state
    step_num = 0
    all_film_frames = []

    while True:
        print(f" \n--- Step {step_num+1} of FiLM Episode ---")

        # Get manual FiLM parameters AFTER diffusion episode completes
        manual_gamma, manual_beta = get_interactive_film_params(env.action_dim, episode_num)

        # Run FiLM episode continuing from current state (no reset)
        reward, done, frame, new_action, final_img_film, final_state_film = run_single_step_with_manual_film(
            env,
            curretnt_action, 
            manual_gamma,
            manual_beta,
            device, 
            args.max_steps,
            current_img,
            current_state
        )

        all_film_frames.append(frame)
        """
        final_state_film: meters
        curretnt_state: meters
        action * 100: meters
        1 action unit = 1 cm in Meta-World
        """

        # Everything in cm 
        print(f" Position after FiLM step: {final_state_film[:3]*100} cm")
        # Show delta position change with 4 decimal places
        delta_pos = (final_state_film[:3] - current_state[:3]) * 100
        print(f" Delta position change (XYZ): {[f'{x:.6f}' for x in delta_pos]} cm")
        # Compute expected position and error
        expected_pos = current_state[:3]*100 + (new_action.squeeze(0).cpu().numpy()[:3])
        error = final_state_film[:3]*100 - expected_pos
        print(f" Expected position after FiLM: {[f'{x:.6f}' for x in expected_pos]}")
        print(f" Error (real - expected): {[f'{x:.6f}' for x in error]}")
        """
        print(f"\n======= Summary ========")
        print(f"  gamma: {manual_gamma.squeeze().cpu().numpy()}")
        print(f"  beta: {manual_beta.squeeze().cpu().numpy()}")
        print(f"  Previous action from diffusion: {last_action.squeeze().cpu().numpy()}")
        print(f"  Last end effector position (XYZ): {final_state[:3]}")
        print(f"  New action (gamma * prev + beta): {new_action.squeeze().cpu().numpy()}")
        print(f"  End effector position after FiLM (XYZ): {final_state_film[:3]}")
        print("="*24 + "\n")
        """

        # Update for next iteration
        curretnt_action = new_action
        current_img = final_img_film
        current_state = final_state_film
        step_num += 1

        if done:
            print(f"FiLM episode finished after {step_num} steps.")
            break
        
        # User input to continue or stop
        user_input = input("\nPress Enter to input new FiLM parameters and continue, or type 'q' to quit this episode: ").strip().lower()
        if user_input == 'q':
            print("Exiting FiLM episode.")
            break

    # Save and log all FiLM steps as a video
    if args.save_video and all_film_frames:
        video_path_film = os.path.join(args.video_dir, f"{args.env_name}_episode{episode_num}_manual_film_steps.mp4")
        with imageio.get_writer(video_path_film, fps=30) as writer:
            for f in all_film_frames:
                f_rot = np.rot90(f, 2)
                writer.append_data(f_rot)

        wandb.log({f"eval/video_manual_film_steps_episode{episode_num+1}": wandb.Video(video_path_film, format="mp4")})


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

    # Run evaluation episodes
    for ep in range(args.episodes):
        run_diffusion_then_manual_film(args, model, env, diff_text_ids, device, ep)
        
    env.close()
    wandb.finish()
    print("[test] Done.")


if __name__ == "__main__":
    main()
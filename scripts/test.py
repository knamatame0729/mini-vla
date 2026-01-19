"""Test VLA Diffusion Policy on Meta-World MT1"""

import os
import argparse
import numpy as np
import torch
import imageio.v2 as imageio
import wandb

from envs.metaworld_env import MetaWorldMT1Wrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from models.llm_film import LLMFiLMGenerator, LLMFiLMWrapper
from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on Meta-World MT1")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vla_diffusion_metaworld_push.pt",
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
        "--film-instruction",
        type=str,
        default="push object to the left side of the table",
        help="Fixed language instruction for FiLM re-conditioning",
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
    parser.add_argument(
        "--use-llm-film",
        action="store_true",
        help="Use LLM to generate FiLM parameters instead of learned MLP",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider for FiLM parameter generation",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (e.g., gpt-4o-mini, claude-3-haiku-20240307)",
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
        
        last_action = action_t.clone()
        action_np = action_t.squeeze(0).cpu().numpy()

        img, state, reward, done, info = env.step(action_np)
        ep_reward += reward
        step += 1
        frames.append(img.copy())

    return ep_reward, step, frames, last_action, img, state


def run_episode_with_llm_film(model, env, llm_film_gen, original_instruction, new_instruction, previous_action, device, max_steps, img_init, state_init):
    """
    Run an episode using LLM-generated FiLM parameters to modulate actions.
    
    The LLM understands:
    - The original instruction the VLA was trained on
    - The action values from VLA
    - The new instruction to adapt to
    """
    img, state = img_init, state_init
    step = 0
    ep_reward = 0.0
    frames = [img.copy()]

    # Generate FiLM parameters using LLM
    with torch.no_grad():
        new_action, gamma, beta = llm_film_gen(action=previous_action, original_instruction=original_instruction, new_instruction=new_instruction)
    
    print(f"\n======= LLM-Generated FiLM parameters ========")
    print(f"  Original instruction: '{original_instruction}'")
    print(f"  New instruction: '{new_instruction}'")
    print(f"  gamma: {gamma.squeeze().cpu().numpy()}")
    print(f"  beta: {beta.squeeze().cpu().numpy()}")
    print(f"  Previous action: {previous_action.squeeze().cpu().numpy()}")
    print(f"  New action: {new_action.squeeze().cpu().numpy()}")

    done = False
    current_action = new_action
    
    while not done and step < max_steps:
        action_np = current_action.squeeze(0).cpu().numpy()

        img, state, reward, done, info = env.step(action_np)
        ep_reward += reward
        step += 1
        frames.append(img.copy())

        if not done and step < max_steps:
            # Re-apply LLM FiLM for next action
            with torch.no_grad():
                current_action, _, _ = llm_film_gen(
                    action=current_action,
                    original_instruction=original_instruction,
                    new_instruction=new_instruction,
                    use_llm=True,
                )

    return ep_reward, step, frames, gamma, beta, current_action, img, state


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
        tags=["evaluation", args.env_name],
    )

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # encode instruction
    diff_tokens = tokenizer.encode(args.instruction)
    diff_text_ids = torch.tensor(diff_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # encode film instruction
    film_tokens = tokenizer.encode(args.film_instruction)
    film_text_ids = torch.tensor(film_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Initialize LLM-FiLM generator if requested
    llm_film_gen = None
    if args.use_llm_film:
        print(f"[test] Using LLM-FiLM with {args.llm_provider}/{args.llm_model}")
        llm_film_gen = LLMFiLMGenerator(
            action_dim=4,  # Meta-World action dim
            llm_provider=args.llm_provider,
            model_name=args.llm_model,
        )

    # environment
    env = MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name="corner1",
    )

    print(f"[test] Meta-World MT1 env: {args.env_name}")
    print(f"[test] state_dim={env.state_dim}, action_dim={env.action_dim}, obs_shape={env.obs_shape}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    # Run evaluation episodes
    for ep in range(args.episodes):
        run_diffusion_then_film_mode(
            args, model, tokenizer, env, 
            diff_text_ids, film_text_ids, device, ep,
            llm_film_gen=llm_film_gen
        )
        
    env.close()
    wandb.finish()
    print("[test] Done.")



def run_diffusion_then_film_mode(args, model, tokenizer, env, diff_text_ids, film_text_ids, device, episode_num, llm_film_gen=None):
    
    print("\n" + "="*60)
    print(f"[Episode {episode_num}]: Starting new test episode with diffusion")
    print("="*60 + "\n")

    # =================== Diffusion Episode ===================
    ep_reward, step, frames, last_action, last_img, last_state = run_episode_with_diffusion(
        model, env, diff_text_ids, device, args.max_steps, args.save_video
        )
    
    print(f"[Diffusion]: reward={ep_reward:.3f}, steps={step}")

    # Save and Log diffusion episode video
    if args.save_video:
        video_path = os.path.join(args.video_dir, f"{args.env_name}_episode{episode_num}_diffusion.mp4")
        with imageio.get_writer(video_path) as writer:
            for f in frames:
                writer.append_data(f)

        wandb.log({f"eval/video_diffusion_episode{episode_num}": wandb.Video(video_path, format="mp4")})



    """Apply FiLM with new prompt to modulate the last action from diffusion episode"""

    print(f"\n[FiLM] Modulate with new prompt: '{args.film_instruction}'")
    
    # =================== FiLM Episode ===================
    if llm_film_gen is not None:
        # Use LLM-generated FiLM parameters
        print("[FiLM] Using LLM to generate gamma/beta...")
        ep_reward, step, frames, gamma, beta, current_action, current_img, current_state = run_episode_with_llm_film(
            model, env, llm_film_gen,
            original_instruction=args.instruction,
            new_instruction=args.film_instruction,
            previous_action=last_action,
            device=device,
            max_steps=args.max_steps,
            img_init=last_img,
            state_init=last_state,
        )
    
    print(f"\n[FiLM]: reward={ep_reward:.3f}, steps={step}")

    # Save and Log FiLM episode video
    if args.save_video:
        video_path = os.path.join(args.video_dir, f"{args.env_name}_film{episode_num:03d}.mp4")
        with imageio.get_writer(video_path) as writer:
            for f in frames:
                writer.append_data(f)

        wandb.log({f"eval/video_film{episode_num}": wandb.Video(video_path, format="mp4")})


if __name__ == "__main__":
    main()

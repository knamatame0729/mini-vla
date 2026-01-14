"""Collect demonstration data from LIBERO benchmark.

LIBERO provides pre-collected expert demonstrations, so this script can either:
1. Load existing demonstrations from LIBERO's dataset
2. Collect new demonstrations using LIBERO's environments (if you have a policy)

Usage:
    # Load pre-collected demos from LIBERO
    python scripts/collect_libero_data.py --task-suite libero_spatial --task-id 0 --num-demos 50

    # List available tasks
    python scripts/collect_libero_data.py --list-tasks --task-suite libero_spatial
"""

import os
import argparse
import numpy as np

from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect LIBERO demonstration data")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_100'],
                        help="LIBERO task suite")
    parser.add_argument("--task-id", type=int, default=0,
                        help="Task index within the suite")
    parser.add_argument("--num-demos", type=int, default=50,
                        help="Number of demonstrations to collect")
    parser.add_argument("--output-path", type=str, default="data/libero_bc.npz",
                        help="Output path for collected data")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List all available tasks in the suite and exit")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Image observation size")
    parser.add_argument("--resize-to", type=int, default=64,
                        help="Resize images to this size for training")
    return parser.parse_args()


def list_available_tasks(task_suite: str):
    """List all tasks in a LIBERO suite."""
    from envs.libero_env import LIBEROEnvWrapper
    
    print(f"\nAvailable tasks in {task_suite}:")
    print("=" * 80)
    tasks = LIBEROEnvWrapper.list_tasks(task_suite)
    for task_id, task_name, instruction in tasks:
        print(f"  [{task_id:2d}] {task_name}")
        print(f"       Instruction: {instruction}")
    print("=" * 80)


def load_libero_demos(args):
    """Load pre-collected demonstrations from LIBERO."""
    from envs.libero_env import get_libero_demo_dataset
    
    print(f"\nLoading LIBERO demonstrations:")
    print(f"  Task suite: {args.task_suite}")
    print(f"  Task ID: {args.task_id}")
    print(f"  Num demos: {args.num_demos}")
    
    data = get_libero_demo_dataset(
        task_suite=args.task_suite,
        task_id=args.task_id,
        num_demos=args.num_demos,
    )
    
    return data


def resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """Resize images to target size."""
    try:
        import cv2
        resized = []
        for img in images:
            resized.append(cv2.resize(img, (target_size, target_size)))
        return np.stack(resized, axis=0)
    except ImportError:
        print("Warning: cv2 not available, skipping resize")
        return images


def main():
    args = parse_args()
    
    # List tasks and exit if requested
    if args.list_tasks:
        list_available_tasks(args.task_suite)
        return
    
    # Load demonstrations
    data = load_libero_demos(args)
    
    print(f"\nLoaded data:")
    print(f"  Images: {data['images'].shape}")
    print(f"  States: {data['states'].shape}")
    print(f"  Actions: {data['actions'].shape}")
    print(f"  Task: {data['task_name']}")
    print(f"  Instruction: {data['instruction']}")
    
    # Resize images if needed
    images = data['images']
    if args.resize_to != images.shape[1]:
        print(f"\nResizing images from {images.shape[1]} to {args.resize_to}...")
        images = resize_images(images, args.resize_to)
    
    # Tokenize instructions
    texts = data['texts']
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)
    
    print(f"\nTokenized instructions:")
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  Max sequence length: {max_len}")
    
    # Save dataset
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(
        args.output_path,
        images=images,
        states=data['states'],
        actions=data['actions'],
        text_ids=text_ids,
        vocab=tokenizer.vocab,
        instruction=data['instruction'],
        task_name=data['task_name'],
        task_suite=args.task_suite,
        task_id=args.task_id,
    )
    
    print(f"\nDataset saved to: {args.output_path}")
    print(f"  Total transitions: {len(images)}")


if __name__ == "__main__":
    main()

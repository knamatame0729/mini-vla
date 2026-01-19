
import numpy as np
import matplotlib.pyplot as plt
from envs.metaworld_env import MetaWorldMT1Wrapper
import wandb
import time

def main():
    env_name = "push-v3"
    seed = 42

    env = MetaWorldMT1Wrapper(
        env_name=env_name,
        seed=seed,
        render_mode="human", 
        camera_name="corner2",
    )

    print(f"[test] state_dim = {env.state_dim}")
    print(f"[test] action_dim = {env.action_dim}")
    print(f"[test] action_space.low  = {env.env.action_space.low}")
    print(f"[test] action_space.high = {env.env.action_space.high}")

    # reset
    img, state, info = env.reset()
    
    # Data storage for plotting
    step_numbers = []
    x_positions = []
    y_positions = []
    z_positions = []
    delta_x = []
    delta_y = []
    delta_z = []
    action_x = []
    action_y = []
    action_z = []
    
    step_count = 0
    prev_state = state.copy()
    
    print("Initial end effector position (XYZ) and Gripper:", state[:4])

    # TEST 1: Apply constant action [1.0, 0.0, 0.0, 0.0]
    
    for i in range(21):
        test_action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, state, _, _, _ = env.step(test_action)
        
        # Store data
        step_numbers.append(step_count)
        x_positions.append(state[0])
        y_positions.append(state[1])
        z_positions.append(state[2])
        delta_x.append(state[0] - prev_state[0])
        delta_y.append(state[1] - prev_state[1])
        delta_z.append(state[2] - prev_state[2])
        action_x.append(test_action[0])
        action_y.append(test_action[1])
        action_z.append(test_action[2])
        
        prev_state = state.copy()
        step_count += 1
        time.sleep(0.1)  # Small delay for visualization

    # TEST 2: Switch to ZERO action [0.0, 0.0, 0.0, 0.0]
    
    for i in range(20):
        zero_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, state, _, _, _ = env.step(zero_action)
        
        # Store data
        step_numbers.append(step_count)
        x_positions.append(state[0])
        y_positions.append(state[1])
        z_positions.append(state[2])
        delta_x.append(state[0] - prev_state[0])
        delta_y.append(state[1] - prev_state[1])
        delta_z.append(state[2] - prev_state[2])
        action_x.append(zero_action[0])
        action_y.append(zero_action[1])
        action_z.append(zero_action[2])
        
        prev_state = state.copy()
        step_count += 1
        time.sleep(0.1) 

    # TEST 3: Apply action [-1.0, -1.0, 0.0, 0.0]
    
    for i in range(20):
        test_action = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        _, state, _, _, _ = env.step(test_action)
        
        # Store data
        step_numbers.append(step_count)
        x_positions.append(state[0])
        y_positions.append(state[1])
        z_positions.append(state[2])
        delta_x.append(state[0] - prev_state[0])
        delta_y.append(state[1] - prev_state[1])
        delta_z.append(state[2] - prev_state[2])
        action_x.append(test_action[0])
        action_y.append(test_action[1])
        action_z.append(test_action[2])
        
        prev_state = state.copy()
        step_count += 1
        time.sleep(0.1) 

    print("Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Position over steps
    ax1 = axes[0, 0]
    ax1.plot(step_numbers, x_positions, 'r-', label='X position', linewidth=2)
    ax1.plot(step_numbers, y_positions, 'g-', label='Y position', linewidth=2)
    ax1.plot(step_numbers, z_positions, 'b-', label='Z position', linewidth=2)
    ax1.axvline(x=20, color='gray', linestyle='--', alpha=0.7, label='Test boundary')
    ax1.axvline(x=40, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(x=60, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('End Effector Position Over Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Delta (velocity) over steps
    ax2 = axes[0, 1]
    ax2.plot(step_numbers, delta_x, 'r-', label='Δx', linewidth=2)
    ax2.plot(step_numbers, delta_y, 'g-', label='Δy', linewidth=2)
    ax2.plot(step_numbers, delta_z, 'b-', label='Δz', linewidth=2)
    ax2.axvline(x=20, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=40, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=60, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Delta Position (m/step)')
    ax2.set_title('Position Change (Velocity) Over Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action commands over steps
    ax3 = axes[1, 0]
    ax3.plot(step_numbers, action_x, 'r-', label='Action X', linewidth=2)
    ax3.plot(step_numbers, action_y, 'g-', label='Action Y', linewidth=2)
    ax3.plot(step_numbers, action_z, 'b-', label='Action Z', linewidth=2)
    ax3.axvline(x=20, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(x=40, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(x=60, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Action Value')
    ax3.set_title('Action Commands Over Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    
    # Plot 5: 2D trajectory (X-Y plane)
    ax5 = axes[1, 1]
    scatter = ax5.scatter(x_positions, y_positions, c=step_numbers, cmap='viridis', s=50, alpha=0.7)
    ax5.plot(x_positions, y_positions, 'k-', alpha=0.3, linewidth=1)
    ax5.scatter(x_positions[0], y_positions[0], c='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax5.scatter(x_positions[-1], y_positions[-1], c='red', s=200, marker='X', label='End', edgecolors='black', linewidths=2)
    ax5.set_xlabel('X Position (m)')
    ax5.set_ylabel('Y Position (m)')
    ax5.set_title('2D Trajectory (X-Y Plane)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Step')
    
    plt.tight_layout()
    plt.show()
    # Save figure in video folder
    fig.savefig("videos/test_analysis_plots.png")

    input("Press Enter to close the environment...")
    env.close()


if __name__ == "__main__":
    main()
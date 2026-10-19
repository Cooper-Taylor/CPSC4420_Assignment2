import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from q_learning import ENV_NAME
from experiment_helpers import RESULTS


VIDEO_EPISODES = 3
VIDEO_SEED = 40000
FPS = 4


if __name__ == "__main__":
    q_path = RESULTS / "final_evaluation" / "best_q_table.npy"
    if not q_path.exists():
        raise FileNotFoundError("Run evaluate_best.py before recording the video.")
    q_table = np.load(q_path)
    output_dir = RESULTS / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "best_agent.mp4"
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    with imageio.get_writer(video_path, fps=FPS, macro_block_size=2) as writer:
        for episode in range(VIDEO_EPISODES):
            state, _ = env.reset(seed=VIDEO_SEED + episode)
            writer.append_data(env.render())
            done = False
            total_reward = 0
            while not done:
                action = int(np.argmax(q_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                writer.append_data(env.render())
                total_reward += reward
                done = terminated or truncated
            # Briefly hold the final frame before the next episode.
            for _ in range(FPS):
                writer.append_data(env.render())
            print(f"Video episode {episode + 1}: reward {total_reward}")

    env.close()
    print(f"Saved {video_path}")

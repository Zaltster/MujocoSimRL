#!/usr/bin/env python3
"""
Train PPO with video recording every N steps
"""

import os
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from drone_catch_env import DroneCatchEnv


class VideoRecorderCallback(BaseCallback):
    """Record videos during training every N steps."""

    def __init__(self, record_freq=1000, video_length=500, verbose=0):
        super(VideoRecorderCallback, self).__init__(verbose)
        self.record_freq = record_freq
        self.video_length = video_length  # Number of steps to record
        self.video_folder = "training_videos"
        os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        # Record video every N steps
        if self.num_timesteps % self.record_freq == 0 and self.num_timesteps > 0:
            print(f"\n🎥 Recording video at step {self.num_timesteps}...")
            self._record_video()

        return True

    def _record_video(self):
        """Record a video of the current policy."""
        # Create a separate environment for rendering
        env = DroneCatchEnv(render_mode="rgb_array")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f"{self.video_folder}/step_{self.num_timesteps:07d}.mp4"
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        obs, info = env.reset()
        frames_recorded = 0
        episode_reward = 0

        for _ in range(self.video_length):
            # Get action from current policy
            action, _states = self.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Render and save frame
            frame = env.render()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                frames_recorded += 1

            if terminated or truncated:
                break

        writer.release()
        env.close()

        print(f"  ✅ Saved video: {video_path}")
        print(f"  Episode reward: {episode_reward:.2f} | Frames: {frames_recorded}")


class VerboseCallback(BaseCallback):
    """Print stats every N steps."""

    def __init__(self, print_freq=1000, verbose=0):
        super(VerboseCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.crashes = 0
        self.catches = 0
        self.misses = 0

    def _on_step(self) -> bool:
        # Collect episode statistics
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    # Track outcomes
                    if ep_reward > 50:
                        self.catches += 1
                    elif ep_reward < -50:
                        self.crashes += 1
                    else:
                        self.misses += 1

        # Print stats every N steps
        if self.num_timesteps % self.print_freq == 0 and self.num_timesteps > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                median_reward = np.median(self.episode_rewards[-100:])
                total = self.catches + self.crashes + self.misses

                print(f"\n📊 Step {self.num_timesteps:,} | Reward: {mean_reward:.1f} (median: {median_reward:.1f}) | "
                      f"Catches: {self.catches}/{total} ({self.catches/max(total,1)*100:.1f}%) | "
                      f"Crashes: {self.crashes} | Misses: {self.misses}")

        return True


def train(record_freq=1000):
    """Train with video recording."""
    print("🚀 Starting PPO Training with Video Recording...")
    print(f"📹 Videos will be saved every {record_freq} steps to training_videos/")

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create environment (no rendering for training - too slow)
    n_envs = 4
    env = SubprocVecEnv([lambda: DroneCatchEnv() for _ in range(n_envs)])

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs/tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="auto"
    )

    # Create callbacks
    video_callback = VideoRecorderCallback(record_freq=record_freq)
    verbose_callback = VerboseCallback(print_freq=1000)

    print("\n🎓 Training started...")
    print("💡 Run: tensorboard --logdir logs/tensorboard/")
    print("📹 Watch videos in: training_videos/\n")

    try:
        model.learn(
            total_timesteps=1000000,
            callback=[verbose_callback, video_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")

    # Save final model
    model.save("models/drone_catch_ppo_final")
    print(f"\n✅ Training complete!")

    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_freq", type=int, default=1000,
                        help="Record video every N steps (default: 1000)")
    args = parser.parse_args()

    train(record_freq=args.record_freq)

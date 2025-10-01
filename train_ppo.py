#!/usr/bin/env python3
"""
Train a PPO agent to catch balls with the drone
Uses Stable-Baselines3
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from drone_catch_env import DroneCatchEnv


def train(args):
    """Train PPO agent."""
    print("🚀 Starting PPO Training...")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Number of parallel environments: {args.n_envs}")

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create vectorized environment (multiple parallel environments for faster training)
    if args.n_envs > 1:
        env = SubprocVecEnv([lambda: DroneCatchEnv() for _ in range(args.n_envs)])
    else:
        env = DummyVecEnv([lambda: DroneCatchEnv()])

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: DroneCatchEnv()])

    # Create callbacks
    # Save model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,  # Adjust for parallel envs
        save_path="./models/",
        name_prefix="drone_catch_ppo"
    )

    # Evaluate model periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/",
        eval_freq=args.eval_freq // args.n_envs,  # Adjust for parallel envs
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Create or load PPO model
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = PPO.load(args.load_model, env=env, tensorboard_log="./logs/tensorboard/")
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            device="auto"  # Use GPU if available
        )

    print("\n📊 Model Architecture:")
    print(model.policy)

    print("\n🎓 Starting training...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    # Save final model
    final_model_path = "models/drone_catch_ppo_final"
    model.save(final_model_path)
    print(f"\n✅ Training complete! Final model saved to {final_model_path}")

    # Clean up
    env.close()
    eval_env.close()

    return model


def test(model_path, n_episodes=5):
    """Test a trained model."""
    print(f"🎮 Testing model: {model_path}")

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = DroneCatchEnv()

    # Run episodes
    total_rewards = []
    catches = 0
    misses = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        print(f"\n🎯 Episode {episode + 1}/{n_episodes}")

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

            if terminated:
                if reward > 5:  # Caught the ball
                    catches += 1
                else:
                    misses += 1

        total_rewards.append(episode_reward)
        print(f"  Episode reward: {episode_reward:.2f} | Steps: {steps}")

    env.close()

    print(f"\n📊 Test Results:")
    print(f"  Average reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"  Catches: {catches} | Misses: {misses}")
    print(f"  Success rate: {catches / n_episodes * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for drone ball catching")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per environment before update (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training (default: 64)")
    parser.add_argument("--save_freq", type=int, default=50000,
                        help="Save model every N steps (default: 50k)")
    parser.add_argument("--eval_freq", type=int, default=25000,
                        help="Evaluate model every N steps (default: 25k)")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load existing model")
    parser.add_argument("--test_model", type=str, default="models/drone_catch_ppo_final",
                        help="Model to test (default: models/drone_catch_ppo_final)")
    parser.add_argument("--test_episodes", type=int, default=5,
                        help="Number of test episodes (default: 5)")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args.test_model, args.test_episodes)


if __name__ == "__main__":
    main()

# 🤖 Reinforcement Learning Training for Drone Ball Catching

## Overview

Train a drone to autonomously catch falling balls using PPO (Proximal Policy Optimization).

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `gymnasium` - RL environment framework
- `stable-baselines3` - PPO implementation
- `mujoco` - Physics simulation
- `torch` - Deep learning backend

## Environment

### Observation Space (16 dimensions)
- Drone position (x, y, z): 3 values
- Drone velocity (vx, vy, vz): 3 values
- Drone orientation (quaternion): 4 values
- Ball position (x, y, z): 3 values
- Ball velocity (vx, vy, vz): 3 values

### Action Space (4 dimensions)
- Continuous actions in [-1, 1]:
  - Thrust (vertical control)
  - Roll (left/right tilt)
  - Pitch (forward/backward tilt)
  - Yaw (rotation)

### Reward Function
- **+10** for catching the ball
- **-1** for missing the ball
- **-0.01 × distance** to ball (encourages pursuit)
- **-0.001 × action_magnitude** (encourages smooth control)
- **-5** for crashing into ground
- **-2** for flying out of bounds

## Training

### Basic Training (1M timesteps)
```bash
python3 train_ppo.py --timesteps 1000000
```

### Advanced Training Options
```bash
python3 train_ppo.py \
  --timesteps 2000000 \
  --n_envs 8 \
  --learning_rate 3e-4 \
  --save_freq 50000 \
  --eval_freq 25000
```

### Parameters
- `--timesteps`: Total training steps (default: 1M)
- `--n_envs`: Parallel environments for faster training (default: 4)
- `--learning_rate`: PPO learning rate (default: 3e-4)
- `--n_steps`: Steps per environment before update (default: 2048)
- `--batch_size`: Batch size for training (default: 64)
- `--save_freq`: Save model every N steps (default: 50k)
- `--eval_freq`: Evaluate every N steps (default: 25k)
- `--load_model`: Continue training from checkpoint

### Resume Training
```bash
python3 train_ppo.py --load_model models/drone_catch_ppo_500000_steps.zip
```

## Testing

### Test Trained Model
```bash
python3 train_ppo.py --mode test --test_model models/drone_catch_ppo_final
```

### Test Best Model
```bash
python3 train_ppo.py --mode test --test_model models/best_model/best_model.zip --test_episodes 10
```

## Monitoring Training

### TensorBoard
Monitor training progress in real-time:
```bash
tensorboard --logdir logs/tensorboard/
```

Then open http://localhost:6006 in your browser.

### Metrics to Watch
- **Episode Reward**: Should increase over time
- **Success Rate**: Percentage of catches vs misses
- **Episode Length**: How long episodes last
- **Policy Loss**: Should stabilize
- **Value Loss**: Should decrease

## Expected Results

### Training Stages
1. **0-100k steps**: Random exploration, ~0% success rate
2. **100k-500k steps**: Learning to move toward ball, ~10-30% success
3. **500k-1M steps**: Improving catching technique, ~50-70% success
4. **1M+ steps**: Fine-tuning, potentially >80% success

### Typical Training Time
- **CPU (4 cores)**: ~8-12 hours for 1M steps
- **GPU**: ~2-4 hours for 1M steps

## File Structure

```
models/
├── drone_catch_ppo_50000_steps.zip    # Checkpoints every 50k steps
├── drone_catch_ppo_100000_steps.zip
├── drone_catch_ppo_final.zip          # Final trained model
└── best_model/
    └── best_model.zip                  # Best performing model

logs/
├── tensorboard/                        # TensorBoard logs
└── evaluations.npz                     # Evaluation metrics
```

## Tips for Better Performance

1. **Tune Hyperparameters**:
   - Increase `n_envs` for faster training (if you have cores)
   - Adjust `learning_rate` if training is unstable
   - Increase `ent_coef` for more exploration

2. **Curriculum Learning**:
   - Start with slower balls
   - Gradually increase difficulty
   - Reduce spawn height over time

3. **Reward Shaping**:
   - Adjust distance penalty coefficient
   - Add penalties for jerky movements
   - Reward hovering near predicted ball landing zone

## Troubleshooting

### Training is Slow
- Increase `--n_envs` (parallel environments)
- Decrease `--n_steps` and `--batch_size`
- Use GPU (PyTorch with CUDA)

### Agent Not Learning
- Check TensorBoard for loss curves
- Increase exploration (`ent_coef`)
- Adjust reward scaling
- Verify environment resets properly

### Memory Issues
- Reduce `--n_envs`
- Reduce `--n_steps`
- Use smaller network architecture

## Next Steps

1. **Vision-Based RL**: Use bottom camera images as observations
2. **Multi-Ball Catching**: Spawn multiple balls simultaneously
3. **Dynamic Difficulty**: Adjust ball speed based on performance
4. **Sim-to-Real**: Transfer learned policy to real drone

## Manual Testing

Play the game manually to understand the task:
```bash
python3 drone_catch_game.py
```

Controls: WASD for movement, Z/X for up/down, Space to hover

Happy Training! 🚁🎾

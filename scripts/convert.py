from stable_baselines3 import PPO
import torch

# Load model from .zip
model = PPO.load("ppo_models/pick_place/models/pick_place_1")

# Get the policy network (this is a torch.nn.Module)
policy = model.policy

# Save the policy as a PyTorch model
torch.save(policy.state_dict(), "policy_weights.pth")

# Optional: convert to TorchScript if you want to use Netron
scripted_policy = torch.jit.script(policy)
scripted_policy.save("policy_model.pt")

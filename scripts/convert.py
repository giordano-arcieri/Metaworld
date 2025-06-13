import torch
import matplotlib.pyplot as plt
import seaborn as sns

policy = torch.jit.load("policy_model.pt", map_location='cpu')

# Instead of policy.mlp_extractor.policy_net[0], grab it via _modules
layer0 = policy.mlp_extractor.policy_net._modules['0']
weights = layer0.weight.detach().cpu().numpy()

# ...existing code...
sns.heatmap(weights, cmap="coolwarm", center=0)
plt.title("Layer 1 Weights")
plt.show()
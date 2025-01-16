import torch

state_dict = torch.load("model_synth.pth")

# Adjust the size of the parameters
state_dict["W_1"] = state_dict["W_1"][:, :4]  # Trim to match current shape
state_dict["W_2"] = state_dict["W_2"][:, :4]
state_dict["W_3"] = state_dict["W_3"][:, :4]
state_dict["W_4"] = state_dict["W_4"][:, :4]

# Save the updated state dictionary
torch.save(state_dict, "model_synth.pth")
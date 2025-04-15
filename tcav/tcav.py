import numpy as np
import torch

def compute_tcav_score(model, cav, layer_output_fn, inputs, target_class):
    """
    For each input, calculate directional derivative in direction of CAV,
    then measure how often it's positive when target_class is true.
    """
    model.eval()
    gradients = []
    targets = []

    for x, y in inputs:
        x = x.unsqueeze(0)
        x.requires_grad = True
        logits = model(x)
        pred = logits[0, target_class]

        model.zero_grad()
        pred.backward(retain_graph=True)

        with torch.no_grad():
            grads = x.grad.flatten().numpy()
            directional_derivative = np.dot(grads, cav)
            gradients.append(directional_derivative)
            targets.append(y.item())

    gradients = np.array(gradients)
    targets = np.array(targets)

    return np.mean((gradients > 0)[targets == target_class])

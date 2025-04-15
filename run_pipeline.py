import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.mlp_model import MLPClassifier
from concepts.cav import train_cav
from tcav.tcav import compute_tcav_score

# Assume: preprocessed tensors for patients, e.g. 1000 x 200 condition features
# X, y = load_preprocessed_data() — skipped
X = torch.randn(1000, 200)  # Fake input for demo
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64)

model = MLPClassifier(input_dim=200)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# === Train Model ===
for epoch in range(10):
    model.train()
    for xb, yb in dataloader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# === Collect Activations ===
def get_activations(data_loader):
    model.eval()
    activations = []
    labels = []
    for xb, yb in data_loader:
        with torch.no_grad():
            z = model.net[0](xb)  # Activations from first layer
            activations.append(z.numpy())
            labels.append(yb.numpy())
    return np.vstack(activations), np.concatenate(labels)

activations, labels = get_activations(dataloader)

# === Create Concept and Random Sets ===
concept_idxs = np.where(labels == 1)[0][:50]
random_idxs = np.where(labels == 0)[0][:50]

concept_cav = train_cav(activations[concept_idxs], activations[random_idxs])

# === TCAV Score ===
sample_inputs = [(X[i], y[i]) for i in range(100)]
tcav_score = compute_tcav_score(model, concept_cav, None, sample_inputs, target_class=1)

print("TCAV score for concept: ", tcav_score)

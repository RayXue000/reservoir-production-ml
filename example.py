"""New synthetic smoke example; no reservoir or well data is used."""
import numpy as np
import torch
from model import SimpleLSTM
from sequence import make_windows

torch.manual_seed(0)
series = np.sin(np.arange(96, dtype=np.float32)[:, None] / 8)
# A chronological boundary is set before building either set of windows.
# An optional scaler must likewise be fitted on training rows only.
x, y = make_windows(series[:72], target_index=0, timestep=4)
x_test, y_test = make_windows(series[72:], target_index=0, timestep=4)
model = SimpleLSTM(input_dim=1, hidden_dim=8, num_layers=1, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(3):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(torch.from_numpy(x)), torch.from_numpy(y))
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    prediction = model(torch.from_numpy(x_test))
assert prediction.shape == torch.from_numpy(y_test).shape
assert torch.isfinite(prediction).all()
print("PASS: sequence shape, training step and held-out forward pass")

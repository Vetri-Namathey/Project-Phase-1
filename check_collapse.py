import numpy as np
import torch
from PIL import Image

import config
from data.fishyscapes_dataset import list_fishyscapes_pairs
from model.twinguard_model import TwinGuardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TwinGuardModel(num_ood_heads=3, ood_seeds=config.OOD_HEAD_SEEDS_3HEAD).to(device)
model.load_state_dict(torch.load(config.CHECKPOINT_3HEAD, map_location=device))
model.eval()

pairs = list_fishyscapes_pairs()[:8]
all_scores = [[] for _ in range(3)]

with torch.no_grad():
    for image_path, _ in pairs:
        image = Image.open(image_path).convert("RGB").resize((config.INPUT_WIDTH, config.INPUT_HEIGHT), Image.BILINEAR)
        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        out = model(image_t)
        for h in range(3):
            all_scores[h].append(out["ood_scores"][:, h].flatten().cpu().numpy())

for h in range(3):
    scores = np.concatenate(all_scores[h])
    print(f"head {h}: mean={scores.mean():.4f}  std={scores.std():.4f}")

from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch
import torchvision

class AudioDS(Dataset):
    def __init__(self, presets_csv_path, spectrograms_path):
        self.presets = pd.read_csv(presets_csv_path)
        self.spectrograms_path = Path(spectrograms_path)

    def __len__(self):
        return len(self.presets)
    
    def __getitem__(self, idx):
        preset = self.presets.iloc[idx]
        preset_name = preset[0]
        preset_path = self.spectrograms_path / f'{preset_name}.png'

        # First, check if the image has already been generated
        if not preset_path.exists():
            raise FileNotFoundError(preset_path)

        try:
            # Load the image
            image = torchvision.io.read_image(preset_path)
        except:
            print('error loading image')
            print(preset_name)

        image = image.float() / 255

        # Take out the first element(preset name) and the last element (empty)
        preset = torch.tensor(preset[1:-1])

        return image, preset
    
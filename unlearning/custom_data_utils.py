from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class CustomDatasetWithMetadata(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with metadata.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.index_to_metadata = self._create_index_to_metadata()

    def _create_index_to_metadata(self):
        """Create a mapping from dataset index to metadata."""
        index_to_metadata = {}
        for i, row in self.img_metadata.iterrows():
            img_path = os.path.join(self.img_dir, row['de_SubfolderName'], row['de_FileName'])
            if os.path.exists(img_path):
                index_to_metadata[i] = row
        return index_to_metadata

    def __len__(self):
        return len(self.index_to_metadata)

    def __getitem__(self, idx):
        if idx in self.index_to_metadata:
            metadata_row = self.index_to_metadata[idx]
            img_path = os.path.join(self.img_dir, metadata_row['de_SubfolderName'], metadata_row['de_FileName'])
            image = Image.open(img_path)

            if self.transform:
                image = self.transform(image)

            # Assuming you have a method to convert metadata to a target/label if needed
            target = self._metadata_to_target(metadata_row)
            print(target)
            return image, target

        else:
            raise IndexError("Index out of range or image file not found for given index.")


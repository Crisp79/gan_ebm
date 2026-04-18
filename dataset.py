import cv2
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        img = cv2.imread(img_path)

        # --- ERROR HANDLING ---
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- TRANSFORMS ---
        if self.transform:
            img = self.transform(image=img)["image"]

        # --- LABEL ---
        label = row["glasses"]

        return img, label
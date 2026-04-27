import torch

from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms


class PokemonDataset(Dataset):
    """
    与 flow-matching 版本完全相同，复制过来方便独立运行。
    """

    def __init__(
        self,
        split: str = "train[:10]",
        img_size: int = 256,
        max_token_len: int = 96,
        vocab=None,
        cache_dir: str = "./data",
    ):
        self.raw_data = load_dataset(
            "svjack/pokemon-blip-captions-en-zh",
            split=split,
            cache_dir=cache_dir,
        )
        self.img_size = img_size
        self.max_token_len = max_token_len
        self.vocab = vocab or {"<pad>": 0, "unknown": 1}

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.raw_data)

    def _tokenize(self, text: str):
        tokens = [self.vocab.get(char, 1) for char in text]
        if len(tokens) < self.max_token_len:
            tokens += [0] * (self.max_token_len - len(tokens))
        else:
            tokens = tokens[: self.max_token_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, index):
        example = self.raw_data[index]
        image = example["image"].convert("RGB")
        image_tensor = self.transform(image)
        caption = example["zh_text"]
        tokens = self._tokenize(caption)
        return image_tensor, tokens

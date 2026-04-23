import os

from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    '''
    Input: IMDB Folder Path
    Output: Review text& Binary label
    '''
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.texts, self.labels = self._load_split()

    def _load_split(self):
        split_dir = os.path.join(self.data_dir, self.split)
        texts = []
        labels = []

        for label_name, label_id in [('neg', 0), ('pos', 1)]:
            label_dir = os.path.join(split_dir, label_name)

            for file_name in sorted(os.listdir(label_dir)):
                if not file_name.endswith('.txt'):
                    continue # txt 아닌 경우에는 Skip

                file_path = os.path.join(label_dir, file_name)

                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())

                labels.append(label_id) #label_id는 0(negative) or 1(positive)

        return texts, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

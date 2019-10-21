from torch.utils.data import Dataset, DataLoader
import json
import numpy as np


class MyDataset(Dataset):

    def __init__(self, file_name, keys):
        file = open(file_name, 'r')
        self.data = json.load(file)
        self.len = len(self.data)
        self.keys = keys

    def __getitem__(self, item):
        cur_data = self.data[item]
        sample = []
        for key in self.keys:
            sample.append(np.array(cur_data[key]))

        return sample

    def __len__(self):
        return self.len


if __name__ == '__main__':
    keys = ["context_idxs", 'ques_idxs', 'context_char_idxs', 'ques_char_idxs', 'y1', 'y2', 'id']
    dataset = MyDataset('data/dev_data.json', keys)

    dataLoader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    for i, data in enumerate(dataLoader):
        print(data[2].size())
        break

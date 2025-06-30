from glob import glob
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
# from .constants import HAM10K_DATA_DIR, DERM7_FOLDER


class DermDataset(Dataset):
    def __init__(self, data_root, df, preprocess=None):
        self.df = df
        self.preprocess = preprocess
        self.paths = np.array(df['path'].values)
        self.paths = np.array([x.replace(data_root, "") for x in self.paths])
        self.labels = np.array(df['y'].values)
        self.data_root = data_root
    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self, index):
        X = Image.open(self.df['path'].iloc[index])
        y = torch.tensor(int(self.df['y'].iloc[index]))
        if self.preprocess:
            X = self.preprocess(X)
        return X,y

# class Derm7ptDataset():
#     def __init__(self, images, base_dir=os.path.join(DERM7_FOLDER, "images"), transform=None,
#                         image_key="derm"):
#         self.images = images
#         self.transform = transform
#         self.base_dir = base_dir
#         self.image_key = image_key
#
#     def __len__(self):
#         return self.images.shape[0]
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         row = self.images.iloc[idx]
#         img_path = os.path.join(self.base_dir, row[self.image_key])
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image


def load_ham_data(params):
    seed = params.get('seed', 0)
    data_root = params['data_root']
    np.random.seed(seed)
    id_to_lesion = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'}

    benign_malignant = {
    'nv': 'benign',
    'mel': 'malignant',
    'bkl': 'benign',
    'bcc': 'malignant',
    'akiec': 'benign',
    'vasc': 'benign',
    'df': 'benign'}

    df = pd.read_csv(os.path.join(data_root,'HAM10000_metadata.csv'))
    all_image_paths = glob(os.path.join(data_root, '*', '*.jpg'))
    id_to_path = {os.path.splitext(os.path.basename(x))[0] : x for x in all_image_paths}

    def path_getter(id):
        if id in id_to_path:
            return id_to_path[id] 
        else:
            return  "-1"
    
    df['path'] = df['image_id'].map(path_getter)
    df = df[df.path != "-1"] 
    df['dx_name'] = df['dx'].map(lambda id: id_to_lesion[id])
    df['benign_or_malignant'] = df["dx"].map(lambda id: benign_malignant[id])
    class_to_idx = {"benign": 0, "malignant": 1}

    df['y'] = df["benign_or_malignant"].map(lambda id: class_to_idx[id])

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    #df = df.groupby("y", group_keys=False).apply(pd.DataFrame.sample, 1000)

    batch_size = params.get('batch_size', 32)
    num_workers = params.get('num_workers', 4)
    transform = params.get('transform', None)
    _, df_val = train_test_split(df, test_size=0.20, random_state=seed, stratify=df["dx"])
    df_train = df[~df.image_id.isin(df_val.image_id)]
    trainset = DermDataset(data_root, df_train, transform)
    valset = DermDataset(data_root, df_val, transform)
    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers)
    
    return dict(train_loader=train_loader, val_loader=val_loader, idx_to_class=idx_to_class, trainset=trainset, valset=valset)


#%%
import random
import pandas as pd
import numpy as np
import os
import re
import glob
import PIL
import shutil
from torchvision import transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore') 
#%%
all_img_list = glob.glob('file_path')
for i in all_img_list:
    os.remove(i)

all_img_list = glob.glob('file_path')
train_df = pd.DataFrame(columns=['img_path','label'])
train_df['img_path'] = all_img_list
train_df['img_path'] = train_df.img_path.str.replace('\\','/')
train_df['label'] = train_df.img_path.apply(lambda x: str(x).split('/')[3])
#%%
train_df, val_df, _, _ = train_test_split(train_df, train_df['label'], test_size=0.3, stratify=train_df['label'], random_state=42)

#%%
val_df[val_df.label == 'label_name']
#%%
for i in val_df.img_path:
    src = i
    dir = i.replace('/train','/val')
    shutil.move(src, dir)

# %%
train_folder_names = glob.glob('file_path')
val_folder_names = glob.glob('file_path')

# %%

for i,v in enumerate(train_folder_names):
    train_folder_names[i] = v.replace('\\','/')
for i,v in enumerate(val_folder_names):
    val_folder_names[i] = v.replace('\\','/')


# %%
for file_path in train_folder_names:
    file_names = os.listdir(file_path)
    i = 0
    for name in file_names:
        src = os.path.join(file_path, name)
        dst = str(i) + '.png'
        dst = os.path.join(file_path,dst)
        os.rename(src, dst)
        i += 1

for file_path in val_folder_names:
    file_names = os.listdir(file_path)
    i = 0
    for name in file_names:
        src = os.path.join(file_path, name)
        dst = str(i) + '.png'
        dst = os.path.join(file_path,dst)
        os.rename(src, dst)
        i += 1

# %%

for file_path in train_folder_names:
    file_names = os.listdir(file_path)
    i = 0
    for name in file_names:
        src = os.path.join(file_path, name)
        dst = str(i) + '.png'
        dst = os.path.join(file_path,dst)
        os.rename(src, dst)
        i += 1
# %%

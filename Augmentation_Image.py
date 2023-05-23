#%%
import random
import pandas as pd
import numpy as np
import os
import re
import glob
import PIL

from torchvision import transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore') 
# %%
CFG = {
    'SEED':41
}
# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정
# %%
all_img_list = glob.glob('file_path')
train_df = pd.DataFrame(columns=['img_path', 'label'])
train_df['img_path'] = all_img_list
train_df['img_path'] = train_df.img_path.str.replace('\\','/')
train_df['label'] = train_df.img_path.apply(lambda x: str(x).split('/')[3])
counts = train_df.groupby('label').count()
print(counts)
# %%
train_df_1 = train_df[train_df.label=='녹오염'].reset_index(drop = True)
train_df_1
transform = T.RandomApply([
        # T.RandomHorizontalFlip(0.75), # 좌우반전, default_p = 0.5
        # T.RandomVerticalFlip(0.75), # 상하반전, depault_p = 0.5
        # T.RandomRotation(degrees=[90, 90]), # image rotate, -각도 ~ + 각도 사이를 rotate
#        T.GaussianBlur(kernel_size=5),
        # T.RandomAffine(45, shear=45) # image rotate + affine
], p = 1.0)
image = PIL.Image.open(train_df_1.img_path[0])
image = transform(image)
image.show()
# %%
for i in range(len(train_df_1), 2*len(train_df_1)):
    j = i - len(train_df_1)
    # j = random.randint(0,len(train_df_1)-1)
    image = PIL.Image.open(train_df_1.img_path[j])
    image = transform(image)
    image.save(f'./wallpaper_relabeling/train/{train_df_1.label[0]}/{i}.png')
    print(i)


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
all_img_list = glob.glob('./도배/train/*/*')
df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = all_img_list
df['img_path'] = df.img_path.str.replace('\\','/')
df['label'] = df.img_path.apply(lambda x: str(x).split('/')[3])
counts = df.groupby('label').count()
counts
# %%
df_1 = df[df.label=='틈새과다'].reset_index(drop = True)
df_1
transform = T.RandomApply([
        # T.RandomHorizontalFlip(1), # 좌우반전, default_p = 0.5
        # T.RandomVerticalFlip(1), # 상하반전, depault_p = 0.5
        T.RandomRotation(15), # image rotate, -각도 ~ + 각도 사이를 rotate
#        T.GaussianBlur(kernel_size=5),
        # T.RandomAffine(15, shear=15) # image rotate + affine
], p = 1.0)
image = PIL.Image.open(df_1.img_path[0])
image = transform(image)
image.show()
# %%
for i in range(20, 50):
    j = random.randint(0,len(df_1)-1)
    image = PIL.Image.open(df_1.img_path[j])
    image = transform(image)
    image.save(f'./도배/train/{df_1.label[0]}/{i}.png')

# %%
df_1.img_path[0]

# %%
len(df_1)
# %%

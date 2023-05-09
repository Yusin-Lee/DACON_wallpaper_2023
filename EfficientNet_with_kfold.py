#%%

import random
import pandas as pd
import numpy as np
import os
import re
import glob
import PIL
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_multi_class_focal_loss.focal_loss import FocalLoss
from torchvision import transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# %%
CFG = {
    'IMG_SIZE':380,
    'EPOCHS':1,
    'LEARNING_RATE':5e-5,
    'BATCH_SIZE':8,
    'SEED':41
}

# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
# %%
data_list = glob.glob('./wallpaper_relabeling/train/*/*')
df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = data_list
df['img_path'] = df.img_path.str.replace('\\','/')
df['label'] = df.img_path.apply(lambda x: str(x).split('/')[3])
df['number'] = df.img_path.apply(lambda x: int(str(x).split('/')[4][:-4]))
df = df.sort_values(by = ['label','number']).reset_index(drop = True)[['img_path','label']]

# %%
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])
df

#%%
for i,v  in enumerate(le.classes_):
    print(v, i)

# %%
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, T = None, mode = None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.mode = mode
        if self.mode == 'train':
            self.T_non, self.T_full, self.T_hor, self.T_ver = T
        else:
            self.T = T
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = PIL.Image.open(img_path)
        
        if self.mode == 'train':
            
            # Full : 곰팡이, 녹오염, 면불량, 오염, 피스, 반점, 울음, 오타공, 훼손
            if self.label_list[index] in [2, 4, 6, 8, 10, 11, 12, 17, 18]:
                image = self.T_full(image)
            
            # 좌우 : 가구수정, 걸레받이수정, 꼬임, 들뜸, 몰딩수정, 석고수정, 창문틀수정, 터짐, 틈새과다
            elif self.label_list[index] in [0, 1, 3, 5, 7, 9, 14, 15, 16] :
                image = self.T_hor(image)

            # 상하 : 이음부불량
            else :
                image = self.T_ver(image)


        else:
            image = self.T(image)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    
# %%
# No aug
train_transform_1 = T.Compose([
                            T.ToTensor(),
                            T.Lambda(lambda x: x[:3]),
                            T.Resize((224,224)),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])

# horizon, vertical
train_transform_2 = T.Compose([
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor(),
                            T.Lambda(lambda x: x[:3]),
                            T.Resize((224,224)),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])

# horizon
train_transform_3 = T.Compose([
                            T.RandomHorizontalFlip(0.5),
                            T.ToTensor(),
                            T.Lambda(lambda x: x[:3]),
                            T.Resize((224,224)),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])

# vertical
train_transform_4 = T.Compose([
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor(),
                            T.Lambda(lambda x: x[:3]),
                            T.Resize((224,224)),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])

test_transform = T.Compose([
                           T.ToTensor(),
                           T.Lambda(lambda x: x[:3]),
                           T.Resize((224,224)),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                           ])

train_transform = [train_transform_1,train_transform_2,train_transform_3,train_transform_4]
#%%
train_dataset = CustomDataset(df.img_path, df.label, T = train_transform, mode = 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0) 

next(iter(train_loader))
#%%
class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model('tf_efficientnet_b4_ns',pretrained = True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
# %%
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = FocalLoss().to(device) # nn.CrossEntropyLoss().to(device)
    
    best_loss = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output,labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')
       
        if scheduler is not None:
            scheduler.step(_val_loss)
            
        if best_loss < _val_score:
            best_loss = _val_score
            best_model = model
            torch.save(best_model.state_dict(), './best_model_score.pt')
            print('best_score!')

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)
            
            output = model(imgs)
            
            loss = criterion(output,labels)
            
            preds += output.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='weighted')
    
    return _val_loss, _val_score
# %%

S_kfold = StratifiedKFold(n_splits = 5, shuffle = False)
model = BaseModel()
model.eval()

optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"])
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

for fold_num, (train_idx, val_idx) in enumerate(S_kfold.split(df['img_path'],df['label'])):
    train_df = df.iloc[train_idx].reset_index(drop = True)
    train_dataset = CustomDataset(train_df.img_path, train_df.label, T = train_transform, mode = 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0) 

    val_df = df.iloc[val_idx].reset_index(drop = True)
    val_dataset = CustomDataset(val_df.img_path,val_df.label, T = test_transform, mode = None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    train(model, optimizer, train_loader, val_loader, None, device)


# %%
infer_model = BaseModel()
infer_model.load_state_dict(torch.load('./best_model_score.pt'))
infer_model.eval()
infer_model = infer_model.to(device)

# %%
test = pd.read_csv('./test_정답.csv',encoding = 'cp949')
test = test[:200]
# %%
test_dataset = CustomDataset(test['img_path'].values, None, test_transform,None)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# %%
def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds

# %%
preds = inference(infer_model, test_loader, device)
#%%
test['pred'] = preds
print(f1_score(test['label'], test['pred'], average='weighted'))
test.to_csv('./testaaa.csv',encoding = 'cp949')


#%%
test = pd.read_csv('./test.csv')
test
# %%
test_dataset = CustomDataset(test['img_path'].values, None, test_transform,None)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# %%
def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds
preds = inference(infer_model, test_loader, device)
# %%
submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds
# submit.loc[submit['label'] == '0', 'label'] = '가구수정'
# submit.loc[submit['label'] == '1', 'label'] = '걸레받이수정'
# submit.loc[submit['label'] == '2', 'label'] = '곰팡이'
# submit.loc[submit['label'] == '3', 'label'] = '꼬임'
# submit.loc[submit['label'] == '4', 'label'] = '녹오염'
# submit.loc[submit['label'] == '5', 'label'] = '들뜸'
# submit.loc[submit['label'] == '6', 'label'] = '면불량'
# submit.loc[submit['label'] == '7', 'label'] = '몰딩수정'
# submit.loc[submit['label'] == '8', 'label'] = '반점'
# submit.loc[submit['label'] == '9', 'label'] = '석고수정'
# submit.loc[submit['label'] == '10', 'label'] = '오염'
# submit.loc[submit['label'] == '11', 'label'] = '오타공'
# submit.loc[submit['label'] == '12', 'label'] = '울음'
# submit.loc[submit['label'] == '13', 'label'] = '이음부불량'
# submit.loc[submit['label'] == '14', 'label'] = '창틀,문틀수정'
# submit.loc[submit['label'] == '15', 'label'] = '터짐'
# submit.loc[submit['label'] == '16', 'label'] = '틈새과다'
# submit.loc[submit['label'] == '17', 'label'] = '피스'
# submit.loc[submit['label'] == '18', 'label'] = '훼손'
submit
#%%
submit.to_csv('./baseline_submit.csv', index=False)
# %%
i,j = np.unique(submit.label,return_counts= True)
print(len(i))
for name, counts in zip(i,j):
    print(name, counts)
# %%

#%%
import random
import pandas as pd
import numpy as np
import os
import re
import glob
import PIL
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import  StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from torch.autograd import Variable
import warnings

warnings.filterwarnings(action='ignore') 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# %%
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':2,
    'LEARNING_RATE':3e-4,
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
name, count = np.unique(df.label,return_counts=True)
for i,j in zip(name, count):
    print(i,j)
# %%
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])
df

#%%
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha]*19)
        self.alpha[18] = 1-alpha
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
# %%
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, T = None, mode = None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.mode = mode

        if self.mode == 'train':
            self.T_full, self.T_hor, self.T_ver = T
        else:
            self.T = T

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = PIL.Image.open(img_path)
        image = np.array(image)[:,:,:3]

        if self.mode == 'train':
            # Full : 곰팡이, 녹오염, 면불량, 오염, 피스, 반점, 석고수정, 울음, 오타공, 훼손
            if self.label_list[index] in [2, 4, 6, 8, 9, 10, 11, 12, 17, 18]:
                image = self.T_full(image=image)
            
            # 좌우 : 가구수정, 걸레받이수정, 꼬임, 들뜸, 몰딩수정, 창문틀수정, 터짐, 틈새과다
            elif self.label_list[index] in [0, 1, 3, 5, 7, 14, 15, 16] :
                image = self.T_hor(image=image)

            # 상하 : 이음부불량
            else :
                image = self.T_ver(image=image)
            image = image['image']
        
        else:
            image = self.T(image = image)
            image = image['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)    
# %%
# horizon, vertical
train_transform_1 = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

# horizon
train_transform_2 = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

# vertical
train_transform_3 = A.Compose([
                            A.VerticalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

train_transform = [train_transform_1,train_transform_2,train_transform_3]
#%%
train_dataset = CustomDataset(df.img_path, df.label, T = train_transform, mode = 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0) 

input, labels = next(iter(train_loader))

print(input.size())
del train_dataset, train_loader, input, labels
#%%
class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model('tf_efficientnet_b3_ns',pretrained = True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
# %%
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    step = 0
    running_loss = 0.

    criterion = FocalLoss(gamma = 2, alpha = 0.25)  #nn.CrossEntropyLoss().to(device)

    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.to(device)
            labels = labels.long().to(device)

            output = model(imgs)
            output = output
            loss = criterion(output, labels)
            
            (loss / accumulation).backward()
            running_loss += loss.item()
            step += 1
            if step % accumulation:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            train_loss.append(running_loss / accumulation)
            running_loss = 0

        _val_loss, _val_score = validation(model,criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')
       
        if scheduler is not None:
            scheduler.step(_val_loss)
            
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.to(device)
            labels = labels.long().to(device)

            output = model(imgs)
            output = output
            loss = criterion(output, labels)
            
            preds += output.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')
    
    return _val_loss, _val_score

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds
# %%

S_kfold = StratifiedKFold(n_splits = 5, shuffle = True)

model = BaseModel()
model.eval()

optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"])
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

batch_size = 32
accumulation = 1

#%%

for fold_num, (train_idx, val_idx) in enumerate(S_kfold.split(df['img_path'],df['label'])):
    train_df = df.iloc[train_idx].reset_index(drop = True)
    train_dataset = CustomDataset(train_df['img_path'].values, train_df['label'].values, T = train_transform, mode = 'train')
    train_loader = DataLoader(train_dataset, batch_size = batch_size//accumulation, shuffle=True, num_workers=0)

    val_df = df.iloc[val_idx].reset_index(drop = True)
    val_dataset = CustomDataset(val_df['img_path'],val_df['label'], T = test_transform, mode = None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size//accumulation, shuffle=False, num_workers=0)

    train(model, optimizer, train_loader, val_loader, None, device)
    torch.save(model.state_dict(), f'./ckp/best_model_score_{fold_num}.pt')
    print(f'{fold_num}_model_save !')    
#%%
input, labels = next(iter(train_loader))
labels
# %%
infer_model = BaseModel()
infer_model.load_state_dict(torch.load('./ckp/best_model_score_3.pt'))
infer_model.eval()
infer_model = infer_model.to(device)
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
submit
#%%
submit.to_csv('./baseline_submit.csv', index=False)
# %%
i,j = np.unique(submit.label,return_counts= True)
print(len(i))
for name, counts in zip(i,j):
    print(name, counts)
# %%

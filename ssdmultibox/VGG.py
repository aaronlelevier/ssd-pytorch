import pdb

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from matplotlib import patches, patheffects
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VGG, vgg16_bn
from torchvision.models.vgg import cfg, model_urls

from ssdmultibox.datasets import (FEATURE_MAPS, Bboxer, TrainPascalDataset,
                                  open_image)
from ssdmultibox.plotting import *

np.set_printoptions(precision=15)


# # Dataset and DataLoader

# In[2]:


dataset = TrainPascalDataset()


# In[3]:


dataloader = DataLoader(dataset, batch_size=4, num_workers=0)


# # Model


model = vgg16_bn(pretrained=True)

feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
blocks = {b:None for b in feat_layers}


for i, f in enumerate(model.features):
    x = model.features[i](x)
    # capture this block
    if i == 22:
        blocks['block4'] = x
    
    # don't call the final max_pool layer b/c we want an output shape of (512, 19, 19)
    if i == 29:
        break

# block6
conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
out6 = F.dropout2d(F.relu(conv6(x)))

# block7
conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
out7 = F.dropout2d(F.relu(conv7(out6)))
blocks['block7'] = out7

# block8
conv8 = nn.Conv2d(1024, 256, kernel_size=1, padding=1)
out8 = F.relu(conv8(out7))

conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
out8_2 = F.relu(conv8_2(out8))
blocks['block8'] = out8_2
out8_2.shape


# block9
conv9 = nn.Conv2d(512, 128, kernel_size=1, padding=1)
out9 = F.relu(conv9(out8_2))

conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
out9_2 = F.relu(conv9_2(out9))
blocks['block9'] = out9_2

# block10
conv10 = nn.Conv2d(256, 128, kernel_size=1, padding=1)
out10 = F.relu(conv10(out9_2))
out10.shape

conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
out10_2 = F.relu(conv10_2(out10))
blocks['block10'] = out10_2

# block11
conv11 = nn.Conv2d(256, 128, kernel_size=1)
out11 = F.relu(conv11(out10_2))

conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
out11_2 = F.relu(conv11_2(out11))
blocks['block11'] = out11_2
out11_2.shape

# check block shapes
for k,v in blocks.items():
    print(k, v.shape)


# OutConv (from fastai) and CustomHead

NUM_CLASSES = len(dataset.categories())
print('num classes:', NUM_CLASSES)

class OutConv(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.oconv1 = nn.Conv2d(nin, 4, 3, padding=1)
        self.oconv2 = nn.Conv2d(nin, NUM_CLASSES, 3, padding=1)
        
    def forward(self, x):
        return [self.flatten_conv(self.oconv1(x)),
                self.flatten_conv(self.oconv2(x))]
    
    def flatten_conv(self, x):
        bs,nf,gx,gy = x.size()
        x = x.permute(0,2,3,1).contiguous()
        return x.view(bs,-1)
    
assert blocks['block10'].shape[1] == 256
out_conv = OutConv(blocks['block10'].shape[1])
out = out_conv(blocks['block10'])
out[0].shape, out[1].shape



bboxer = Bboxer()

class CustomHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv512 = OutConv(512)
        self.conv1024 = OutConv(1024)
        self.conv256 = OutConv(256)
        
        self.block_map = {
            'block4': self.conv512,
            'block7': self.conv1024,
            'block8': self.conv512,
            'block9': self.conv256,
            'block10': self.conv256,
            'block11': self.conv256
        }
        
    def forward(self, blocks):
        ret = []
        aspect_ratio_count = 6
        for k,v in blocks.items():
            ret.append([self.block_map[k](v) for _ in range(aspect_ratio_count)])
        return ret
        
custom_head = CustomHead()
out = custom_head(blocks)
len(out)


# In[48]:


len(out[0])


# In[50]:


len(out[0][0])


# size per block match

# In[84]:


for i in range(6):
    print(i, out[i][0][0].shape, out[i][0][1].shape)


# In[87]:


for i in range(6):
    print(i, gt_bbs[i][0].shape, gt_cats[i][0].shape)


# size along the 1d match, should all be the same. this is the per aspect-ratio with the same
# feature cell map size

# In[81]:


for i in range(6):
    print(i, out[0][i][0].shape, out[0][i][1].shape)


# In[88]:


for i in range(6):
    print(i, gt_bbs[0][i].shape, gt_cats[0][i].shape)


# # NEXT: combine CustomHead with Model and DataLoader

# In[43]:





# In[44]:


def get_anchor_centers(anc_grid):
    anc_offset = 1/(anc_grid*2)
    anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)
    anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)
    anc_ctrs = np.stack([anc_x,anc_y], axis=1)
    return anc_ctrs

anc_ctrs = get_anchor_centers(3)

anc_x = anc_ctrs[:,0]
anc_y = anc_ctrs[:,1]

plt.scatter(anc_x, anc_y)
plt.xlim(0, 1)
plt.ylim(0, 1);


# In[45]:


anc_ctrs


# In[46]:


# square anchors
anc_grid = 3
anc_sizes = np.array([[1/anc_grid,1/anc_grid] for i in range(anc_grid*anc_grid)])
anc_sizes # they're all this size


# In[47]:


# feature map cell default hw
sk = 1/anc_grid
w = 2. * sk
h = 1. * sk
np.reshape(np.repeat([w, h], anc_grid*anc_grid), (2,-1)).T


# In[48]:


def hw2corners(center, hw):
    return np.concatenate([center-hw/2, center+hw/2], axis=1)

anchor_corners = hw2corners(anc_ctrs, anc_sizes)
anchor_corners


# In[49]:


SIZE = 300
ax = show_img(im)
for i, bbox in enumerate(anchor_corners):
    if i == 0:
        draw_rect(ax, bbox*SIZE, edgecolor='red')


# In[50]:


ax = show_img(im)
xy = np.array([1/3, 1/3]) * SIZE
width = height = 1/3 * SIZE

# grid
for i, bbox in enumerate(anchor_corners):
    draw_rect(ax, bbox*SIZE, edgecolor='grey')

# focus default box
ax.add_patch(patches.Rectangle(xy, width, height, fill=False, edgecolor='red', lw=2))


# In[51]:


width


# In[52]:


np.sqrt(.2*.2+1)


# `sk` is based upon the `grid_size`
# 
# ```python
# grid_size = 5
# hw = (1,1)
# sk = grid_size / hw
# sk ~ 1
# ```

# In[53]:


sk = 1/3


# In[54]:


# aspect ration
ar = np.array([
    (1., 1.), 
    (2., 1.),
    (3., 1.),
    (1., 2.),
    (1., 3.),
    (np.sqrt(sk*sk+1), 1.)
])
ar


# In[55]:


# width, heigh
wh = sk * ar
wh


# In[56]:


center = anc_ctrs
hw = wh[0]
anc_corners = np.concatenate([center-hw/2, center+hw/2], axis=1)
ax = show_img(im)
# grid
for i, bbox in enumerate(anchor_corners):
    draw_rect(ax, bbox*SIZE, edgecolor='white')
    
for j, hw in enumerate(wh):
    x1 = anc_ctrs[:,0]-hw[0]/2
    y1 = anc_ctrs[:,1]-hw[1]/2
    x2 = anc_ctrs[:,0]+hw[0]/2
    y2 = anc_ctrs[:,1]+hw[1]/2
    anc_corners = np.stack([x1,y1,x2,y2], axis=1)
    for i, bbox in enumerate(anc_corners):
        # focus default box
        if i == 4:
            xy = np.array(bbox[:2])*SIZE
            width = (bbox[2] - bbox[0])*SIZE
            height = (bbox[3] - bbox[1])*SIZE
            ax.add_patch(patches.Rectangle(xy, width, height, fill=False, edgecolor=f'C{j}', lw=2))
            break


# In[57]:


grid_size = 5
sk = 1 / 5
anc_ctrs = dataset.bboxer.anchor_centers(grid_size)
anc_corners = dataset.bboxer.anchor_corners(grid_size)

# image
ann = dataset.get_annotations()[17]
im = open_image(ann['image_path'])
ax = show_img(cv2.resize(im, (SIZE, SIZE)))

# grid
for i, bbox in enumerate(anc_corners):
    draw_rect(ax, bbox*SIZE, edgecolor='white')

for j, hw in enumerate(dataset.bboxer.aspect_ratios(grid_size)):
    hw = hw * sk
    x1 = anc_ctrs[:,0]-hw[0]/2
    y1 = anc_ctrs[:,1]-hw[1]/2
    x2 = anc_ctrs[:,0]+hw[0]/2
    y2 = anc_ctrs[:,1]+hw[1]/2
    anc_corners = np.stack([x1,y1,x2,y2], axis=1)
    for i, bbox in enumerate(anc_corners):
        # focus default box
        if i == 7:
            xy = np.array(bbox[:2])*SIZE
            width = (bbox[2] - bbox[0])*SIZE
            height = (bbox[3] - bbox[1])*SIZE
            ax.add_patch(patches.Rectangle(xy, width, height, fill=False, edgecolor=f'C{j}', lw=2))
            break


# In[75]:


bboxer = dataset.bboxer
grid_size = 4
sk = 1 / grid_size
anc_ctrs = dataset.bboxer.anchor_centers(grid_size)
anc_corners = dataset.bboxer.anchor_corners(grid_size)

# image
ann = dataset.get_annotations()[17]
im = open_image(ann['image_path'])
ax = show_img(cv2.resize(im, (SIZE, SIZE)))

for i, bbox in enumerate(bboxer.anchor_corners(grid_size)):
    if i == 5:
        color = 'red'
    else:
        color = 'white'
    draw_rect(ax, bbox*SIZE, edgecolor=color)

horse_bb = dataset.bboxer.pascal_bbs(np.expand_dims(np.reshape(gt_bbs[3][0], (-1,4))[7], axis=0)).squeeze()
draw_rect(ax, horse_bb, edgecolor='black')
person_bb = dataset.bboxer.pascal_bbs(np.expand_dims(np.reshape(gt_bbs[3][0], (-1,4))[11], axis=0)).squeeze()
draw_rect(ax, person_bb, edgecolor='black')


# In[63]:


image_id, chw_im, gt_bbs, gt_cats = item


# In[67]:


# 5x5 grid_size, 1:1 aspect ratio
gt_cats[3][0]


# In[71]:


np.where(gt_cats[3][0] != 20)[0]


# In[77]:


bbs = ann['bbs']
bbs = np.array(bbs)
bbs


# In[78]:


im.shape


# In[79]:


bboxer.get_bbs_area(bbs, im)


# In[82]:


bbs[:,2:]


# In[85]:


95*138


# In[86]:


364*480


# In[87]:


(95*138) / (364*480)


# In[88]:


(314*259) / (364*480)


# In[89]:


bboxer.get_ancb_area(grid_size)


# In[91]:


bboxer.get_iou(bbs, im).argmax(1)


# In[93]:


gt_overlap, gt_idx = bboxer.get_gt_overlap_and_idx(bbs, im)
gt_overlap, gt_idx


# In[97]:


gt_overlap.max(0)


# In[99]:


gt_idx[5]


# In[100]:


gt_overlap > .5


# In[101]:


cats = np.array(ann['cats'])
cats


# In[103]:


gt_bbs, gt_cats = bboxer.get_gt_bbs_and_cats(bbs, cats, im)
gt_bbs, gt_cats


# In[106]:


np.reshape(gt_bbs, (-1,4))[5]


# In[107]:


bbs


# In[109]:


bboxer.scaled_fastai_bbs(bbs, im) * SIZE


# In[110]:


cats


# In[111]:


dataset.categories()[12]


# In[84]:


im.shape


# In[ ]:


cv2.resize(im, (SIZE, SIZE))


# In[ ]:


bboxer.anchor_corners(grid_size=4)


# In[ ]:


bbs


# In[ ]:


bbs = bboxer.scaled_fastai_bbs(ann['bbs'], im)
bbs


# In[ ]:


bbs_count = grid_size*grid_size
bbs_count


# In[ ]:


bbs16 = np.reshape(np.tile(bbs, bbs_count), (-1,bbs_count,4))
print(bbs16.shape)
bbs16


# In[ ]:


anchor_corners = bboxer.anchor_corners(grid_size)
anchor_corners


# In[ ]:


bbs


# In[ ]:


np.maximum(anchor_corners[:,:2], bbs16[:,:,:2])


# In[ ]:


np.minimum(anchor_corners[:,2:], bbs16[:,:,2:])


# In[ ]:


intersect = np.minimum(
    np.maximum(anchor_corners[:,:2], bbs16[:,:,:2]) - \
    np.minimum(anchor_corners[:,2:], bbs16[:,:,2:]), 0)
intersect


# In[ ]:


ret = intersect[:,:,0] * intersect[:,:,1]
ret


# In[ ]:


ret.argmax(1)


# In[ ]:


def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def box_sz(b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
    return inter / union


# In[ ]:


np.minimum(1, 0)


# In[ ]:


t_bbs = torch.tensor(bbs)
t_anchor_corners = torch.tensor(anchor_corners)
inter = intersect(t_bbs, t_anchor_corners)
inter


# In[ ]:


inter.max(1)


# In[ ]:


iou = jaccard(t_bbs, t_anchor_corners)
iou


# In[ ]:


iou.max(1)


# In[ ]:


iou.max(0)


# In[ ]:


anchor_corners

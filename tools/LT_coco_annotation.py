import os
import numpy as np
import json
import torch
from pycocotools.coco import COCO
file="/home//project/NLT-multi-label-classification/dataset/coco/original/img_id.txt"
file_content=np.loadtxt(file,dtype=str)
path = "/hdd8//dataset/coco/annotations/instances_train2017.json"
coco = COCO(path)
cat2cat = dict()
dataset=dict()
total= [0] * 80
def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c
for cat in coco.cats.keys():
    cat2cat[cat] = len(cat2cat)
for i in range(len(file_content)):
    img_id=int(file_content[i])
    ann_ids = coco.getAnnIds(imgIds=img_id)
    target = coco.loadAnns(ann_ids)
    output = torch.zeros((3, 80), dtype=torch.long)
    for obj in target:
        if obj['area'] < 32 * 32:
            output[0][cat2cat[obj['category_id']]] = 1
        elif obj['area'] < 96 * 96:
            output[1][cat2cat[obj['category_id']]] = 1
        else:
            output[2][cat2cat[obj['category_id']]] = 1
    target = output
    target = target.max(dim=0)[0].tolist()
    img_id="0" * (12 - len(str(img_id))) + str(img_id)+".jpg"
    dataset[img_id]=target
json_object = json.dumps(dataset, indent=1)
with open("/hdd8//dataset/coco/annotations/LT_coco_temp_train.json", "w") as outfile:
    outfile.write(json_object)
    # print (img_id)
    # print (target)
    # total=list_add(target,total)

# print (total)
# images=row_data["images"]
# annotations=row_data["annotations"]
# images_new=[]
# annotations_new=[]
# for i in range(len(annotations)):
#     annotations_line=annotations[i]
#     if(str(annotations_line["image_id"]) in file_content):
#         annotations_new.append(annotations_line)
# file_content_new=[]
# for i in range(len(file_content)):
#     image_name=file_content[i]
#     file_content_new.append ("0" * (12 - len(image_name)) + image_name)
# for i in range(len(images)):
#     images_line=images[i]
#     if(str(images_line["file_name"].split(".")[0]) in file_content_new):
#         images_new.append(images_line)
# row_data["images"]=images_new
# row_data["annotations"]=annotations_new
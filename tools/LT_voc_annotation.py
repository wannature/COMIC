import os
import numpy as np
import json
annotation_path ="/hdd8//dataset/voc2012/VOCdevkit/VOC2012/ImageSets/Main"
class_path="/hdd8//dataset/voc2012/class.txt"
image_id_path="/home//project/LT-Multi-Classification/appendix/VOCdevkit/longtail2012/img_id.txt"
class_cate=np.loadtxt(class_path,delimiter=",",dtype=str)
class_cate[19]="tvmonitor"
image_id=np.loadtxt(image_id_path,dtype=str)
class_file_name=[]
annatation_files=[]
number_result=[0]*len(class_cate)
for i in range(len(class_cate)):
    class_file_name.append(class_cate[i]+"_trainval.txt")
    annatation_files.append(np.loadtxt(os.path.join(annotation_path,class_file_name[i]), dtype=str))
result={}
for i in range(len(image_id)):
    key=image_id[i]+".jpg"
    value=[0]*len(class_cate)
    for j in range(len(class_cate)):
        index=np.where(annatation_files[j][:,0]==image_id[i])
        if(annatation_files[j][index][0][1]=="1"):
            value[j]=1
            number_result[j]+=1
    result[key]=value
json_object = json.dumps(result, indent=4)
with open("/hdd8//dataset/voc2012/annotation/LT_voc_train.json", "w") as outfile:
    outfile.write(json_object)
# print(number_result)
# number_json={}
# for i in range(len(class_cate)):
#     number_json[class_cate[i]]=number_result[i]
# print(number_json)



# annotation_path="/hdd8//dataset/voc2007/VOCdevkit/VOC2007/ImageSets/Main"
# class_path="./class.txt"
# class_cate=np.loadtxt(class_path,delimiter=",",dtype=str)
# class_cate[19]="tvmonitor"
# image_id=np.loadtxt(os.path.join(annotation_path,"aeroplane_test.txt"),dtype=str)[:,0]
# class_file_name=[]
# annatation_files=[]
# number_result=[0]*len(class_cate)
# for i in range(len(class_cate)):
#     class_file_name.append(class_cate[i]+"_test.txt")
#     annatation_files.append(np.loadtxt(os.path.join(annotation_path,class_file_name[i]), dtype=str))
# result={}
# for i in range(len(image_id)):
#     key=image_id[i]+".jpg"
#     value=[0]*len(class_cate)
#     for j in range(len(class_cate)):
#         # index=np.where(annatation_files[j][:,0]==image_id[i])
#         if(annatation_files[j][i][1]=="1"):
#             value[j]=1
#             number_result[j]+=1
#     result[key]=value
# json_object = json.dumps(result, indent=4)
# with open("/hdd8//dataset/coco/annotations/new_label_instances_test.json", "w") as outfile:
#     outfile.write(json_object)
# print(number_result)
# number_json={}
# for i in range(len(class_cate)):
#     number_json[class_cate[i]]=number_result[i]
# print(number_json)
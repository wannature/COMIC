import torch
distribution_path="/home//project/NLT-multi-label-classification/dataset/coco/longtail2017/distribution.txt"
with open(distribution_path) as f:
    for line in f:
        list_temp=line.replace(" ","").split(",")
list_distribution=list(map(int,list_temp))
num = sum(list_distribution)
prob = [i/num for i in list_distribution]
prob = torch.FloatTensor(prob)
max_prob = prob.max().item()
prob = prob / max_prob
weight = - prob.log() + 1
print (weight)
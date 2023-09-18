#baseline
python baseline.py --cfg /home//project/COMIC/config/coco/resnet50_baseline.yaml
#our method


#coco
#balanced
python plt_mlc_main.py --cfg /home//project/COMIC/config/coco/resnet50_plt.yaml
#head
python plt_mlc_head_tail.py --cfg /home//project/COMIC/config/coco/resnet50_head.yaml
#tail
python plt_mlc_head_tail.py --cfg /home//project/COMIC/config/coco/resnet50_tail.yaml


#voc
python plt_mlc_main.py --cfg /home//project/COMIC/config/voc/voc_resnet50_plt.yaml
#head
python plt_mlc_head_tail.py --cfg /home//project/COMIC/config/voc/resnet50_head.yaml
#tail
python plt_mlc_head_tail.py --cfg /home//project/COMIC/config/voc/resnet50_tail.yaml
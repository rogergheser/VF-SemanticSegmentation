python datasets/prepare_coco_stuff_164k_sem_seg.py datasets/coco
python tools/mask_cls_collect.py datasets/coco/stuffthingmaps/train2017 datasets/coco/stuffthingmaps_detectron2/train2017_base_label_count.pkl
python tools/mask_cls_collect.py datasets/coco/stuffthingmaps/val2017 datasets/coco/stuffthingmaps_detectron2/val2017_label_count.pkl
echo "Processed COCO"

python datasets/prepare_voc_sem_seg.py datasets/VOC2012
python tools/mask_cls_collect.py datasets/VOC2012/annotations_detectron2/train_base datasets/VOC2012/annotations_detectron2/train_base_label_count.json
python tools/mask_cls_collect.py datasets/VOC2012/annotations_detectron2/val datasets/VOC2012/annotations_detectron2/val_label_count.json
echo "Processed VOC2012"

python datasets/prepare_ade20k_sem_seg.py
echo "Processed ADE20k"

python datasets/prepare_pcontext_sem_seg.py --ori_root_dir datasets/pcontext --save_dir datasets/pcontext-59
echo "Processed Pascal Context"

python datasets/prepare_ade20k_full_sem_seg.py
echo "Processed ADE20k FULL"

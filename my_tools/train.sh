nohup python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 8 -b 512 --fp16  --resume -c /opt/tiger/minist/YOLOX/YOLOX_outputs/yolox_voc_s/last_epoch_ckpt.pth 2>&1 &



nohup python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_l.py \
-d 4 -b 96 --fp16  \
-c /opt/tiger/minist/YOLOX/weight/yolox_l.pth -o 2>&1 &
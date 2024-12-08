#!/bin/bash
if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=30
fi

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_out/'
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='configs/swin_patch8_window64_512.yaml'
fi

if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='datasets/archive'
fi

if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.05
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=512
fi

if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=24
fi

echo "start test model"
python test.py --dataset mine --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --is_savenii

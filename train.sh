LOGNAME=log$(date +"%Y%m%d_%H%M%S")

HOST='127.0.0.1'
PORT='12345'

NUM_GPU=4

python train.py \
--cfg './configs/r50.json' \
--data_dir '/mnt/lustre/share/txwu/data/Seq-DeepFake' \
--dataset_name 'facial_attributes' \
--val_epoch 10 \
--model_save_epoch 5 \
--manual_seed 777 \
--launcher pytorch \
--rank 0 \
--log_name ${LOGNAME} \
--dist-url tcp://${HOST}:${PORT} \
--world_size $NUM_GPU \
--results_dir './results'
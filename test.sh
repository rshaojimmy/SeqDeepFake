LOGNAME='YOUR-LOGNAME-HERE'

HOST='127.0.0.1'
PORT='12345'

NUM_GPU=1

python test.py \
--cfg './configs/r50.json' \
--data_dir 'YOUR-DATASET-ROOT-HERE' \
--dataset_name 'facial_attributes' \
--test_type 'adaptive' \
--launcher pytorch \
--rank 0 \
--log_name ${LOGNAME} \
--dist-url tcp://${HOST}:${PORT} \
--world_size $NUM_GPU \
--results_dir './results'

# dataset_name: facial_components facial_attributes
# test_type: fixed adaptive


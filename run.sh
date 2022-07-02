
REMAKR=${1:-"resolution_384_same_mtk"}

LOG_PATH=${2:-"${REMAKR}/log"}
RESULT_PATH=${3:-"${REMAKR}/results"}
CHECKPOINT_PATH=${4:-"${REMAKR}/checkpoint"}

python main.py --mode train \
               --scope unet \
               --name_data em \
               --dir_data datasets \
               --dir_log ${LOG_PATH} \
               --dir_result ${RESULT_PATH} \
               --dir_checkpoint ${CHECKPOINT_PATH} \
               --nch_ker 16 \
               --ny_in 384 \
               --nx_in 384 \
               --ny_load 384 \
               --nx_load 384 \
               --ny_out 384 \
               --nx_out 384 \
               --nch_in 3 \
               --nch_out 3 \
               --gpu_ids 0
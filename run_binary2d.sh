
REMAKR=${1:-"resolution_384_same_mtk_binary"}

LOG_PATH=${2:-"${REMAKR}/log"}
RESULT_PATH=${3:-"${REMAKR}/results"}
CHECKPOINT_PATH=${4:-"${REMAKR}/checkpoint"}

python binary2d_main.py --mode train \
               --scope unet \
               --name_data em \
               --dir_data data_mtk \
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
               --gpu_ids 0 \
               --pretrained-model /home/shenxuan/project/UNet/pytorch_k_unet_master/resolution_384_same_mtk_input3channel/checkpoint/unet/em/model_epoch0500.pth \
               --alpha 25 \
               --percent 0.5

# test on XGaze-train

CUDA_VISIBLE_DEVICES=7 python main.py \
            --mode normal \
            --mask_network True \
            --data_name 'XGaze-train' \
            --data_dir '/mnt/train' \
            --is_train False \
            --shuffle False \
            --batch_size 30 \
            --print_freq 10 \
            --pre_trained_model_path '/home/jqin/GazeEstimation/ckpt/epoch_24_ckpt.pth.tar' \
            --ckpt_dir '<your path for saving test results>' \
            --log_dir '<your path for log (no use for testing)>'

# test on XGaze-test

# CUDA_VISIBLE_DEVICES=7 python main.py \
#             --mode normal \
#             --mask_network True \
#             --data_name 'XGaze-test' \
#             --data_dir '/mnt/test' \
#             --is_train False \
#             --shuffle False \
#             --batch_size 30 \
#             --print_freq 10 \
#             --pre_trained_model_path '/home/jqin/GazeEstimation/ckpt/epoch_24_ckpt.pth.tar' \
#             --ckpt_dir '/home/jqin/GazeEstimation/ckpt/' \
#             --log_dir '/home/jqin/GazeEstimation/logs'


# # Test on EYEDIAP
# CUDA_VISIBLE_DEVICES=8 python main.py \
#             --mode normal \
#             --mask_network True \
#             --num_workers 5 \
#             --data_name 'EYEDIAP' \
#             --data_dir '<>' \
#             --is_train False \
#             --batch_size 100 \
#             --print_freq 200 \
#             --pre_trained_model_path '<>' \
#             --ckpt_dir '<>' \
#             --log_dir '<>'



# # test on GazeCapture.h5
# CUDA_VISIBLE_DEVICES=2 python main.py \
#             --mode normal \
#             --mask_network False \
#             --data_name 'GazeCapture' \
#             --data_dir '<>/GazeCapture.h5' \
#             --is_train False \
#             --shuffle False \
#             --batch_size 250 \
#             --print_freq 200 \
#             --pre_trained_model_path '<>' \
#             --ckpt_dir '<>' \
#             --log_dir '<>'
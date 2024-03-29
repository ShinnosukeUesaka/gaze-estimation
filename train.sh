CUDA_VISIBLE_DEVICES=4,5 python main.py \
             --mode normal \
             --mask_network True \
             --data_name 'xgaze' \
             --data_dir '/work/s-uesaka/xgaze_224_augmented/train' \
             --train_test_split_file '/home/s-uesaka/gaze-estimation-xgaze/my_split.json'\
             --is_train True \
             --shuffle True \
             --batch_size 20 \
             --epochs 50 \
             --save_epoch 1 \
             --print_freq 100 \
             --ckpt_dir '/home/s-uesaka/gaze-estimation-xgaze/temp_output' \
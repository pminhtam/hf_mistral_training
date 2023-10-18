python colab_train.py \
    --data data/epoch2 \
    --lr 5e-6 \
    --lr_scheduler cosine \
    --epochs 2 \
    --lora_r 256 \
    --neftune 5 \
	--warmup_step 100 \
    --batch_size 128 \
    --exp_name 5e-6_cosine_r256_epoch2_warmup100_batchsize128

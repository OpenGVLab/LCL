torchrun --nproc_per_node $NPROC_PER_NODE  -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data '/path/to/laion' \
    --dataset-type webdataset \
    --imagenet-val '/path/to/ImageNet/val' \
    --warmup 2000 \
    --batch-size $BATCH_SIZE \
    --epochs 32 \
    --workers 10 \
    --model LCL_ViT-B-32_laion \
    --name LCL_ViT-B-32_laion \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --use-interleaved-wrapper \
    --interleaved-context-length 128 \
    --num-img-token 49 \
    --img-first-prob 0.5 \
    --lcl-generation-loss-weight 1.0 \
    --lcl-contrastive-loss-weight 0.1 \
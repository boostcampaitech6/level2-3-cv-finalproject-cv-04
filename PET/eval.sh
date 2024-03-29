CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --dataset_file="SHA" \
    --resume="./best_checkpoint.pth" \
    --vis_dir="" \
    --transformer_method="pet_crt" \
    --pet_method="only_encoder_v2"
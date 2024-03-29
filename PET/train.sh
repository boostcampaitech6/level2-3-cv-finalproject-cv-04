BACKBONE="vgg16_bn"
OUTPUT="test"

python main.py \
    --lr=0.0001 \
    --backbone=${BACKBONE} \
    --position_embedding="sine" \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --eos_coef=0.5 \
    --dec_layers=1 \
    --hidden_dim=256 \
    --dim_feedforward=512 \
    --nheads=8 \
    --dropout=0.0 \
    --epochs=1500 \
    --batch_size=8 \
    --dataset_file="SHA" \
    --eval_freq=1 \
    --output_dir=${OUTPUT} \
    --transformer_method="pet_crt" \
    --pet_method="only_encoder_v2"
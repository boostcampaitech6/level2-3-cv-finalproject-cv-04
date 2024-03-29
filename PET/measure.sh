# vgg16_bn, efficient_b0
# python measure_inference_time.py \
python measure_modulewise_inference_time.py \
    --backbone="vgg16_bn" \
    --position_embedding="sine" \
    --dec_layers=2 \
    --dim_feedforward=512 \
    --hidden_dim=256 \
    --dropout=0.0 \
    --nheads=8 \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --eos_coef=0.5 \
    --dataset_file="SHA" \
    --num_workers=8 \
    --resume="./outputs/SHA/Transformer_87_Only_Encoder_Final_V2_VGG19_BN/best_checkpoint.pth" \
    --repetitions=10 \
    --transformer_method="final_manual_attention" \
    --pet_method="final_v2" \
    --measure_mode="total"
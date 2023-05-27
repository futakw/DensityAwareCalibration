# extract features and save knn scores
python3 extract_feature_and_knn_score.py \
    --data_root data \
    --batch_size 128 \
    --dataset cifar10 \
    --model_name resnet18 \
    --model_path classifier_ckpts/cifar10/resnet18.pth \
    --save_only_ood_scores True \
    --top_k 50 \
    --test_data_type "natural" \
        "gaussian_noise" \
        "shot_noise" \
        "speckle_noise" \
        "impulse_noise" \
        "defocus_blur" \
        "gaussian_blur" \
        "motion_blur" \
        "zoom_blur" \
        "snow" \
        "fog" \
        "brightness" \
        "contrast" \
        "elastic_transform" \
        "pixelate" \
        "jpeg_compression" \
        "spatter" \
        "saturate" \
        "frost"
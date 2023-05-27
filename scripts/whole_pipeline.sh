# extract features and save knn scores
python3 extract_feature_and_knn_score.py \
    --data_root data \
    --batch_size 128 \
    --dataset cifar10 \
    --model_name resnet18 \
    --model_path classifier_ckpts/cifar10/resnet18.pth \
    --save_only_ood_scores True \
    --top_k 50 \
    --test_data_type natural gaussian_noise

# run DAC (compare with ETS or SPL)
combination_method=ETS # ETS or SPL
python3 run_experiment.py \
    --dataset cifar10 \
    --save_outputs_dir outputs/cifar10/resnet18 \
    --save_outputs_type ood_score \
    --ood_scoring_layers_list maxpool layer1 layer2 layer3 layer4 logits \
    --combination_method $combination_method \
    --test_data_type natural_1 gaussian_noise_3 gaussian_noise_5
python3 run_experiment.py \
    --dataset cifar10 \
    --model_name resnet18 \
    --save_outputs_dir outputs/cifar10/resnet18 \
    --knn_k 50 \
    --ood_scoring_layers_list maxpool \
    --combination_method SPL \
    --test_data_type natural_1 gaussian_noise_3 gaussian_noise_5
  
# --ood_scoring_layers_list maxpool layer1 layer2 layer3 layer4 logits \

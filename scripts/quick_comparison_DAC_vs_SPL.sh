python3 run_experiment.py \
    --dataset cifar10 \
    --save_outputs_dir outputs/cifar10/resnet18 \
    --save_outputs_type ood_score \
    --ood_scoring_layers_list maxpool layer1 layer2 layer3 layer4 logits \
    --combination_method SPL \
    --test_data_type natural_1 gaussian_noise_3
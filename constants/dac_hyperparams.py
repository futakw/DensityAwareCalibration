hyperparams = {
    "cifar10": {
        "num_classes": 10,
        "knn_k": 50,
        "train_data_ratio": 1.0
    },
    "cifar100": {
        "num_classes": 100,
        "knn_k": 200,
        "train_data_ratio": 1.0
    },
    "imagenet":{
        "num_classes": 1000,
        "knn_k": 10, # if using 1% of ImageNet data, use 10
        "train_data_ratio": 0.1
    }
}
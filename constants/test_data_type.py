corruption_names = [
    "natural",
    "gaussian_noise",
    "shot_noise",
    "speckle_noise",
    "impulse_noise",
    "defocus_blur",
    "gaussian_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "spatter",
    "saturate",
    "frost",
]

severity = [1,2,3,4,5]

all_test_data_type = [
    f"{c_name}_{s}"
    for c_name in corruption_names
    for s in severity
]
def get_hyperparams():
    return {
        "batch_size": 128,
        "patch_size": 7,
        "epochs": 10,
        "learning_rate": 1e-3,
        "num_patches":48,
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 3,
        "num_classes": 12,
        "img_size": (28,84),
        "data_path": "./data",
        "seq_len": 4,
    }


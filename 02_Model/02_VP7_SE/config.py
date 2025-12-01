config = {
    "model_name": "/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D",
    "batch_size": {
        "VP7": 64,
    },
    "lr": 2e-5,
    "epochs": 50,
    "max_length": {
        "VP7": 360,
    },
    "device": "cuda",
    "save_dir": "output_8M",
    "early_stopping_patience": 5,
    "best_model_path": "best_model",
    "f1_csv": "test_f1_by_fold.csv",
}

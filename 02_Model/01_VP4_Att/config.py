config = {
    "model_name": "/media/server/DATA/lijiajun/project/Rotavirus/02_Model/esm_model/esm2_t6_8M_UR50D",  # 可选：esm2_t6_8M_UR50D, esm2_t33_650M_UR50D 等
    "batch_size": {
        "VP4": 18,
    },
    "lr": 2e-5,
    "epochs": 50,
    "max_length": {
        "VP4": 850,
    },
    "device": "cuda",
    "save_dir": "output_8M",
    "early_stopping_patience": 5,
    "best_model_path": "best_model",
    "f1_csv": "test_f1_by_fold.csv",

}

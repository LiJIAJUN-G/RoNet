config = {
    "model_name": "/media/server/DATA/lijiajun/project/Rotavirus/02_Model/00_Pretraining/esm_pretrain_output/final_model",
    "batch_size": {
        "VP7": 16,
    },
    "lr": 2e-5,
    "epochs": 32,
    "max_length": {
        "VP7": 360,
    },
    "device": "cuda",
    "save_dir": "output_8M_pretraining",
    "early_stopping_patience": 8,
    "best_model_path": "best_model",
    "f1_csv": "test_f1_by_fold.csv",
}

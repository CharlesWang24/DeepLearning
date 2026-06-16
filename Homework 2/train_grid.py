from homework.train import train

jobs = [
    # Linear Classifier - try different learning rates
    {
        "model_name": "linear",
        "num_epoch": 50,
        "lr": 1e-2,
    },
    {
        "model_name": "linear",
        "num_epoch": 50,
        "lr": 1e-3,
    },
    {
        "model_name": "linear",
        "num_epoch": 100,
        "lr": 1e-3,
    },
    
    # MLP - try different learning rates and epochs
    {
        "model_name": "mlp",
        "num_epoch": 50,
        "lr": 1e-2,
    },
    {
        "model_name": "mlp",
        "num_epoch": 50,
        "lr": 1e-3,
    },
    {
        "model_name": "mlp",
        "num_epoch": 100,
        "lr": 1e-3,
    },
    
    # Deep MLP - may need more epochs or different LR
    {
        "model_name": "mlp_deep",
        "num_epoch": 50,
        "lr": 1e-3,
    },
    {
        "model_name": "mlp_deep",
        "num_epoch": 100,
        "lr": 1e-3,
    },
    {
        "model_name": "mlp_deep",
        "num_epoch": 50,
        "lr": 5e-4,
    },
    
    # Deep Residual - residuals often train faster
    {
        "model_name": "mlp_deep_residual",
        "num_epoch": 50,
        "lr": 1e-3,
    },
    {
        "model_name": "mlp_deep_residual",
        "num_epoch": 100,
        "lr": 1e-3,
    },
    {
        "model_name": "mlp_deep_residual",
        "num_epoch": 50,
        "lr": 1e-2,
    },
]

for i, params in enumerate(jobs):
    print(f"\n{'='*60}")
    print(f"Job {i+1}/{len(jobs)}: {params['model_name']} | epochs={params['num_epoch']} | lr={params['lr']}")
    print(f"{'='*60}\n")
    train(**params)

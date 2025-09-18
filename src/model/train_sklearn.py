# src/model/train_sklearn.py
import os, argparse, numpy as np, torch, joblib, wandb
from torch.utils.data import TensorDataset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def read(data_dir: str, split: str) -> TensorDataset:
    x, y = torch.load(os.path.join(data_dir, split + ".pt"))
    return TensorDataset(x.to(torch.float32), y.to(torch.float32).view(-1, 1))

def to_numpy(ds: TensorDataset):
    x, y = ds.tensors
    return x.numpy(), y.numpy().ravel()

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str)
parser.add_argument('--n_estimators', type=int, default=800)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--max_depth', type=int, default=3)
parser.add_argument('--subsample', type=float, default=0.9)
parser.add_argument('--min_samples_leaf', type=int, default=1)
parser.add_argument('--validation_fraction', type=float, default=0.2)
parser.add_argument('--n_iter_no_change', type=int, default=20)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

if not args.IdExecution:
    args.IdExecution = "testing console"

with wandb.init(
    project=os.getenv("WANDB_PROJECT", "Exp_Analitica"),
    entity=os.getenv("WANDB_ENTITY"),
    name=f"Train GBR ExecId-{args.IdExecution}",
    job_type="train-model-sklearn",
    config=vars(args),
) as run:
    # datos
    data_art = run.use_artifact("california-housing-preprocess:latest", type="dataset")
    data_dir = data_art.download()
    train = read(data_dir, "training")
    val   = read(data_dir, "validation")
    test  = read(data_dir, "test")
    Xtr, ytr = to_numpy(train)
    Xval, yval = to_numpy(val)
    Xte, yte = to_numpy(test)

    # entrenamos con early stopping interno (validation_fraction sobre train+val)
    X = np.concatenate([Xtr, Xval], axis=0)
    y = np.concatenate([ytr, yval], axis=0)
    gbr = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_samples_leaf=args.min_samples_leaf,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        random_state=args.random_state,
    )
    gbr.fit(X, y)

    # métricas finales
    yval_pred = gbr.predict(Xval)
    yte_pred  = gbr.predict(Xte)
    val_rmse = float(np.sqrt(mean_squared_error(yval, yval_pred)))
    val_mae  = float(mean_absolute_error(yval, yval_pred))
    val_r2   = float(r2_score(yval, yval_pred))
    te_rmse  = float(np.sqrt(mean_squared_error(yte, yte_pred)))
    te_mae   = float(mean_absolute_error(yte, yte_pred))
    te_r2    = float(r2_score(yte, yte_pred))

    wandb.log({
        "val/rmse": val_rmse, "val/mae": val_mae, "val/r2": val_r2,
        "test/rmse": te_rmse, "test/mae": te_mae, "test/r2": te_r2
    })
    run.summary.update({
        "val/rmse": val_rmse, "val/mae": val_mae, "val/r2": val_r2,
        "test/rmse": te_rmse, "test/mae": te_mae, "test/r2": te_r2
    })

    # guardar y publicar artifact
    os.makedirs("models", exist_ok=True)
    path = "models/gbreg.joblib"
    joblib.dump(gbr, path)

    art = wandb.Artifact(
        "california-gbr", type="model",
        description="GradientBoostingRegressor on California Housing",
        metadata=gbr.get_params()
    )
    art.add_file(path)
    run.log_artifact(art)
# 
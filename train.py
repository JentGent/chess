import torch
from torch.utils.data import DataLoader
import model
import shutil
from pathlib import Path
from data import Incremental

BATCH_SIZE = 32 # number of samples in each batch
VAL_SIZE = 32 # number of samples to use for validation
SAVE_PERIOD = 100 # checkpoint every SAVE_PERIOD batches
ROOT_DIR = Path(__file__).parent

def latest_ckp(best = False):
    ckp_dir = ROOT_DIR / "checkpoints"
    existing_ckps = [int(x.stem.split("_")[-1]) for x in ckp_dir.glob("best_*.tar" if best else "*.tar") if x.is_file()]
    return -1 if not existing_ckps else max(existing_ckps)

def save_ckp(state, is_best):
    ckp_dir = ROOT_DIR / "checkpoints"
    next_ckp = latest_ckp() + 1
    checkpoint_path = ckp_dir / (f"best_{next_ckp}.tar" if is_best else f"checkpoint_{next_ckp}.tar")
    torch.save(state, checkpoint_path)

def load_ckp(ckp_path, model, optimizer):
    ckp = torch.load(ckp_path)
    model.load_state_dict(ckp["state_dict"])
    optimizer.load_state_dict(ckp["optimizer"])
    return model, optimizer, ckp["loss"], ckp["val"]

def validate(model: model.Evaluate, val_loader: DataLoader, loss_fn):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        for i, (states, evals) in enumerate(val_loader):
            running_loss += loss_fn(model(states.to(DEVICE)), evals.view(VAL_SIZE, 1).to(DEVICE)).item()
            if i > VAL_SIZE: return running_loss / VAL_SIZE
        return running_loss / VAL_SIZE

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    train_size = 12954834
    test_size = 1000273
    train_dataset = Incremental(ROOT_DIR / "dataset" / "chessData.csv", num_samples = train_size)
    test_dataset = Incremental(ROOT_DIR / "dataset" / "random_evals.csv", num_samples = test_size)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = VAL_SIZE, shuffle = False)

    evaluate = model.Evaluate().to(DEVICE)
    optimizer = torch.optim.Adam(evaluate.parameters(), lr = 0.001)

    latest_best = latest_ckp(best = True)
    best_val = float("inf") if latest_best == -1 else load_ckp(ROOT_DIR / "checkpoints" / f"best_{latest_best}.tar", evaluate, optimizer)[3]
    
    loss_fn = torch.nn.MSELoss()
    running_loss = 0
    for i, (states, evals) in enumerate(train_loader):
        if i % SAVE_PERIOD == 0:
            print(f"Batch {i}")
        optimizer.zero_grad()
        predictions = evaluate(states.to(DEVICE))
        loss = loss_fn(predictions, evals.view(BATCH_SIZE, 1).to(DEVICE))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % SAVE_PERIOD == 0:
            loss = running_loss / SAVE_PERIOD
            val = validate(evaluate, test_loader, loss_fn)
            print(f"Loss: {loss}")
            print(f"Val: {val}")
            save_ckp({ "state_dict": evaluate.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss, "val": val }, val < best_val + 0.1)
            
            if val < best_val + 0.1: best_val = val
            else: load_ckp(ROOT_DIR / "checkpoints" / f"best_{latest_ckp(best = True)}.tar", evaluate, optimizer)
            running_loss = 0

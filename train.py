import torch
from torch.utils.data import DataLoader
import model
import shutil
from pathlib import Path
from data import Incremental

BATCH_SIZE = 32
VAL_SIZE = 8
SAVE_PERIOD = 100 # checkpoint every SAVE_PERIOD batches
ROOT_DIR = Path(__file__).parent

def latest_ckp(best = False):
    ckp_dir = ROOT_DIR / "checkpoints" / "best" if best else ROOT_DIR / "checkpoints"
    existing_ckps = [int(x.stem.split("_")[-1]) for x in ckp_dir.glob("checkpoint_*.tar") if x.is_file()]
    return -1 if not existing_ckps else max(existing_ckps)

def save_ckp(state, is_best):
    ckp_dir = ROOT_DIR / "checkpoints"
    next_ckp = latest_ckp() + 1
    checkpoint_path = ckp_dir / f"checkpoint_{next_ckp}.tar"
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, ckp_dir / "best" / f"checkpoint_{next_ckp}.tar")

def load_ckp(ckp_path, model, optimizer):
    ckp = torch.load(ckp_path)
    model.load_state_dict(ckp["state_dict"])
    optimizer.load_state_dict(ckp["optimizer"])
    return model, optimizer

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
    best_loss = float("inf") if latest_best == -1 else load_ckp(ROOT_DIR / "checkpoints" / "best" / f"checkpoint_{latest_best}.tar", evaluate, optimizer)
    
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
            last_loss = running_loss / SAVE_PERIOD
            print(last_loss)
            save_ckp({ "state_dict": evaluate.state_dict(), "optimizer": optimizer.state_dict(), "loss": last_loss }, last_loss < best_loss)
            
            if last_loss < best_loss: best_loss = last_loss
            else: load_ckp(ROOT_DIR / "checkpoints" / "best" / f"checkpoint_{latest_ckp(best = True)}.tar", evaluate, optimizer)
            running_loss = 0

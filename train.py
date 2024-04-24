import torch
from torch.utils.data import DataLoader
import model
from pathlib import Path
import data
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 64 # number of samples in each batch
SAVE_PERIOD = 1000 # checkpoint every SAVE_PERIOD batches
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
    return model, optimizer, ckp["loss"]

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    writer = SummaryWriter()

    train_dataset = data.DatasetFromPath(ROOT_DIR / "dataset" / "chessData.csv")
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False)

    evaluate = model.Evaluate().to(DEVICE)
    optimizer = torch.optim.Adam(evaluate.parameters(), lr = 0.001)

    latest_best = latest_ckp(best = True)
    best_loss = float("inf") if latest_best == -1 else load_ckp(ROOT_DIR / "checkpoints" / f"best_{latest_best}.tar", evaluate, optimizer)[2]
    
    loss_fn = torch.nn.MSELoss()
    running_loss = 0

    for epoch in range(1):
        for i, (states, evals) in enumerate(train_loader):
            if i % SAVE_PERIOD == 0:
                print(f"Epoch {epoch}, batch {i}")
            optimizer.zero_grad()
            predictions = evaluate(states.to(DEVICE))
            loss = loss_fn(predictions, evals.view(BATCH_SIZE, 1).to(DEVICE))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % SAVE_PERIOD == 0:
                loss = running_loss / SAVE_PERIOD / BATCH_SIZE
                writer.add_scalar("loss", loss, epoch * len(train_loader) + i)
                print(f"Loss: {loss}")
                save_ckp({ "state_dict": evaluate.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss }, loss < best_loss)
                
                if loss < best_loss: best_loss = loss
                elif loss > best_loss + 0.01:
                    print("Warm restart.")
                    load_ckp(ROOT_DIR / "checkpoints" / f"best_{latest_ckp(best = True)}.tar", evaluate, optimizer)
                running_loss = 0

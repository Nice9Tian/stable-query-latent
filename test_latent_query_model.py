import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from latent_query_model import LatentQueryFlatRegressor


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_script_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


class H5LatentQueryDataset(Dataset):
    def __init__(self, h5_path, indices, target_mean=None, target_std=None):
        self.h5_path = Path(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_mean = target_mean
        self.target_std = target_std
        self.handle = None

    def _h5(self):
        if self.handle is None:
            self.handle = h5py.File(self.h5_path, "r")
        return self.handle

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = int(self.indices[item])
        h5 = self._h5()
        tokens = torch.from_numpy(h5["inputs"][row]).float()
        key_padding_mask = torch.from_numpy(h5["key_padding_mask"][row]).bool()
        target = torch.from_numpy(h5["targets"][row]).float()

        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std

        return tokens, key_padding_mask, target

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def parse_query_sizes(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def make_split(sample_count, test_ratio, seed):
    generator = np.random.default_rng(seed)
    indices = generator.permutation(sample_count)
    test_count = max(1, int(round(sample_count * test_ratio)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    if len(train_indices) == 0:
        raise ValueError("Train split is empty; reduce --test-ratio.")
    return train_indices, test_indices


def evaluate(model, loader, target_mean, target_std, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    criterion = torch.nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for tokens, key_padding_mask, targets in loader:
            tokens = tokens.to(device)
            key_padding_mask = key_padding_mask.to(device)
            targets = targets.to(device)

            predictions = model(tokens, key_padding_mask=key_padding_mask)
            total_loss += criterion(predictions, targets).item()

            raw_predictions = predictions * target_std.to(device) + target_mean.to(device)
            raw_targets = targets * target_std.to(device) + target_mean.to(device)
            total_mae += torch.abs(raw_predictions - raw_targets).sum().item()
            total_count += targets.numel()

    return total_loss / total_count, total_mae / total_count


def train_and_test(
    h5_path,
    epochs,
    batch_size,
    learning_rate,
    min_learning_rate,
    test_ratio,
    seed,
    hidden_dim,
    flat_dim,
    query_sizes,
    num_heads,
    dropout,
    device_name,
    model_out,
    history_txt,
):
    torch.manual_seed(seed)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    h5_path = resolve_script_path(h5_path)
    if model_out:
        model_out = resolve_script_path(model_out)
    if history_txt:
        history_txt = resolve_script_path(history_txt)

    with h5py.File(h5_path, "r") as h5:
        sample_count = h5["inputs"].shape[0]
        input_dim = h5["inputs"].shape[2]
        output_dim = h5["targets"].shape[1]
        all_targets = torch.from_numpy(h5["targets"][:]).float()

    train_indices, test_indices = make_split(sample_count, test_ratio, seed)
    train_targets = all_targets[torch.from_numpy(train_indices)]
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0).clamp_min(1e-6)

    train_dataset = H5LatentQueryDataset(h5_path, train_indices, target_mean, target_std)
    test_dataset = H5LatentQueryDataset(h5_path, test_indices, target_mean, target_std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LatentQueryFlatRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        flat_dim=flat_dim,
        query_sizes=query_sizes,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_learning_rate,
    )
    criterion = torch.nn.MSELoss()

    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_item_count = 0

            for tokens, key_padding_mask, targets in train_loader:
                tokens = tokens.to(device)
                key_padding_mask = key_padding_mask.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                predictions = model(tokens, key_padding_mask=key_padding_mask)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * targets.numel()
                train_item_count += targets.numel()

            train_mse = train_loss_sum / train_item_count
            test_mse, test_mae = evaluate(model, test_loader, target_mean, target_std, device)
            current_lr = scheduler.get_last_lr()[0]
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                }
            )
            print(
                f"epoch={epoch:03d} "
                f"lr={current_lr:.8f} "
                f"train_mse={train_mse:.6f} "
                f"test_mse={test_mse:.6f} "
                f"test_mae_raw={test_mae:.6f}"
            )
            scheduler.step()
    finally:
        train_dataset.close()
        test_dataset.close()

    if model_out:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "flat_dim": flat_dim,
            "query_sizes": query_sizes,
            "num_heads": num_heads,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "history": history,
        }
        torch.save(checkpoint, model_out)

    if history_txt and history:
        history_txt.parent.mkdir(parents=True, exist_ok=True)
        with history_txt.open("w", encoding="utf-8") as file:
            file.write("epoch\tlearning_rate\ttrain_mse\ttest_mse\ttest_mae\n")
            for row in history:
                file.write(
                    f"{row['epoch']}\t"
                    f"{row['learning_rate']:.10g}\t"
                    f"{row['train_mse']:.10g}\t"
                    f"{row['test_mse']:.10g}\t"
                    f"{row['test_mae']:.10g}\n"
                )

    return history[-1] if history else None


def main():
    parser = argparse.ArgumentParser(
        description="Train/test latent_query_model as a score regressor from HDF5."
    )
    parser.add_argument("--input-h5", default="benchmark_sentence_latent_query.h5")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--flat-dim", type=int, default=128)
    parser.add_argument(
        "--query-sizes",
        type=parse_query_sizes,
        default=(32, 16, 8),
        help="Comma-separated latent query counts, for example 32,16,8.",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-out", default="latent_query_benchmark_test.pt")
    parser.add_argument("--history-txt", default="latent_query_training_history.txt")
    args = parser.parse_args()

    final_metrics = train_and_test(
        h5_path=args.input_h5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        test_ratio=args.test_ratio,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        flat_dim=args.flat_dim,
        query_sizes=args.query_sizes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device_name=args.device,
        model_out=args.model_out,
        history_txt=args.history_txt,
    )
    if final_metrics:
        print(
            "final: "
            f"train_mse={final_metrics['train_mse']:.6f}, "
            f"test_mse={final_metrics['test_mse']:.6f}, "
            f"test_mae_raw={final_metrics['test_mae']:.6f}"
        )
        print(f"saved model checkpoint: {args.model_out}")
        print(f"saved training history txt: {args.history_txt}")


if __name__ == "__main__":
    main()

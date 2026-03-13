import argparse
from pathlib import Path
import torch


def convert_partition(root: Path, partition: str, mask_mode: str, seed: int = 42):
    src = root / f"{partition}_processed_target" / "data.pt"
    if mask_mode == "aligned":
        raise ValueError("aligned does not need conversion")
    dst = root / f"{partition}_processed_target_{mask_mode}" / "data.pt"

    if not src.exists():
        raise FileNotFoundError(f"source dataset not found: {src}")

    data_list = torch.load(src)
    for g in data_list:
        if not hasattr(g, "x") or g.x is None or g.x.shape[1] < 769:
            continue
        if mask_mode == "all_ones":
            g.x[:, 768] = 1.0
        elif mask_mode == "random":
            sid = int(g.sample_id[0].item()) if hasattr(g, "sample_id") else 0
            gen = torch.Generator(device=g.x.device if g.x.is_cuda else 'cpu')
            gen.manual_seed(seed + sid)
            rnd = torch.bernoulli(torch.full((g.x.shape[0],), 0.5, device=g.x.device), generator=gen)
            g.x[:, 768] = rnd
        else:
            raise ValueError(f"unsupported mask_mode={mask_mode}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, dst)
    print(f"saved: {dst} (graphs={len(data_list)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mask_mode", choices=["all_ones", "random"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partitions", type=str, default="train,val,test")
    args = parser.parse_args()

    parts = [p.strip() for p in args.partitions.split(",") if p.strip()]
    for part in parts:
        convert_partition(args.root, part, args.mask_mode, args.seed)


if __name__ == "__main__":
    main()

import argparse
import os

import torch


def merge_metadata(pt_paths, out_path):
    merged = []
    for p in pt_paths:
        obj = torch.load(p, weights_only=True)  # 若你的 torch 不支持 weights_only 就删掉这个参数
        md = obj["metadata"]
        if not isinstance(md, list):
            raise TypeError(f"{p}: metadata is not a list, got {type(md)}")
        merged.extend(md)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"metadata": merged}, out_path)
    print(f"Saved to {out_path}. Total items: {len(merged)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_paths", nargs="+", help="paths to .pt files")
    parser.add_argument("--out", required=True, help="output .pt path")
    args = parser.parse_args()

    merge_metadata(args.pt_paths, args.out)
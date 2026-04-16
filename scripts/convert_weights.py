import torch
import argparse
from titans.utils import TitansConfig, build_model

def convert_titans_weights(checkpoint_path, save_path, variant="MAC"):
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Define config based on state_dict if possible, or use standard
    # This is a template script to be updated when official weights drop
    cfg = TitansConfig.small(variant=variant)
    model = build_model(cfg)
    
    # Custom mapping logic here
    # model.load_state_dict(state_dict)
    
    print(f"Saving converted weights to {save_path}...")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--variant", type=str, default="MAC")
    args = parser.parse_args()
    convert_titans_weights(args.checkpoint, args.output, args.variant)

"""Checkpoint helpers for local files."""
import os

import torch


def require_checkpoint(path: str) -> str:
    """The given checkpoint file, or a clear exit naming it."""
    if not path or not os.path.isfile(path):
        raise SystemExit(f'No checkpoint file at {path!r}')
    return path


def checkpoint_file(args) -> str:
    """DIR/NAME.ckpt: where THIS training run saves (and, if it exists at startup, fully resumes from)."""
    return os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.ckpt')


def load_checkpoint_state(checkpoint_path: str, model, strict: bool = True) -> None:
    """Load a Lightning checkpoint's ``state_dict`` into ``model``."""
    # STRICT by default. test_cross_model relies on a codec whose shapes do not match the
    # checkpoint raising here: the target is then recorded as skipped_incompatible.
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')
    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint = torch.load(checkpoint_file, device)
    model.load_state_dict(checkpoint['state_dict'], strict)

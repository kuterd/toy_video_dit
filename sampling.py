import torch
import utils
from tqdm import tqdm


def sample(model, x_ids, frames, width, height, patch_w, patch_h, schedule, device, batch_size = 8):
    
    x = torch.rand(batch_size, frames, 1, width, height).to(device)
    
    n_steps = 100
    sigmas = schedule.sample_sigmas(n_steps)
    
    for (sigma_prev, sigma) in tqdm(list(zip(sigmas[1:], sigmas[:-1]))):
        with torch.no_grad():
            _x = utils.patchify(x, patch_w, patch_h)
            
            _ids = x_ids[:batch_size]
            _sigma = torch.full((x.shape[0],), sigma, device=device)
    
            _preds = model(_x, _ids, _sigma).detach()
            preds = utils.unpatchify(_preds, patch_w, patch_h, height, width, 1)
    
        mix_factor = (sigma - sigma_prev) / 10
        x = x + preds * mix_factor

    return x


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

def load_checkpoint(path, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    in_state_dict = ckpt["model_state_dict"]
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt["model_state_dict"] = out_state_dict
    return ckpt

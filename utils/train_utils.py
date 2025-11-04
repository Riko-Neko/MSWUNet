import torch


def safe_load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    new_state_dict = {}
    mismatch = False
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                print(f"[\033[33mWarn\033[0m] Shape mismatch for '{k}': "
                      f"model {model_dict[k].shape} vs checkpoint {v.shape}. Reinitializing.")
                mismatch = True
        else:
            print(f"[\033[33mWarn\033[0m] Key '{k}' not in model, skipping.")
            mismatch = True
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)
    print("[\033[32mInfo\033[0m] State dict loaded with mismatched keys reinitialized.")
    return mismatch


def load_optimizer_selectively(optimizer, opt_state_dict):
    """
    Selectively load optimizer state by parameter order instead of id(param),
    avoiding cross-session id mismatch.
    """
    if 'state' not in opt_state_dict or 'param_groups' not in opt_state_dict:
        print("[\033[33mWarn\033[0m] Invalid optimizer state dict, skipping.")
        return
    old_state = list(opt_state_dict['state'].values())
    new_state = {}
    matched_count = 0
    total_old = len(old_state)
    new_params = [p for g in optimizer.param_groups for p in g['params']]

    for i, p in enumerate(new_params):
        amsgrad = any(g.get('amsgrad', False) and p in g['params'] for g in optimizer.param_groups)
        if i < total_old:
            old_p_state = old_state[i]
            if ('exp_avg' in old_p_state and
                    isinstance(old_p_state['exp_avg'], torch.Tensor) and
                    old_p_state['exp_avg'].shape == p.shape):
                new_entry = {k: (v.to(p.device) if torch.is_tensor(v) else v)
                             for k, v in old_p_state.items()}
                matched_count += 1
            else:
                new_entry = {'step': torch.tensor(0., device=p.device), 'exp_avg': torch.zeros_like(p, device=p.device),
                             'exp_avg_sq': torch.zeros_like(p, device=p.device)}
                if amsgrad:
                    new_entry['max_exp_avg_sq'] = torch.zeros_like(p, device=p.device)
        else:
            new_entry = {'step': torch.tensor(0., device=p.device), 'exp_avg': torch.zeros_like(p, device=p.device),
                         'exp_avg_sq': torch.zeros_like(p, device=p.device)}
            if amsgrad:
                new_entry['max_exp_avg_sq'] = torch.zeros_like(p, device=p.device)

        new_state[p] = new_entry
    optimizer.state = new_state
    for group, saved_group in zip(optimizer.param_groups, opt_state_dict['param_groups']):
        for k, v in saved_group.items():
            if k != 'params':
                group[k] = v
    if matched_count != total_old:
        print(f"[\033[33mWarn\033[0m] Optimizer state not fully loaded: "
              f"{matched_count}/{total_old} matched, {len(new_params) - matched_count} reinitialized.")
    else:
        print(f"[\033[32mInfo\033[0m] Optimizer state loaded. ({matched_count}/{total_old} matched)")

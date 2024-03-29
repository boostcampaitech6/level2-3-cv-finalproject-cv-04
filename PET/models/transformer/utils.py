from einops import rearrange
"""
Transformer window-rize tools
"""


def window_partition(x, window_size_h, window_size_w, mode="basic"):
    B, C, H, W = x.shape
    
    if mode == "basic":  # [L B' C]
        x = rearrange(x, "B C (win_h h) (win_w w) -> (h w) (B win_h win_w) C",
                win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    elif mode == "poolformer":  # [B' C H' W']
        x = rearrange(x, "B C (win_h h) (win_w w) -> (B win_h win_w) C h w",
                win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    elif mode == "batch_first":  # [B' L C]
        x = rearrange(x, "B C (win_h h) (win_w w) -> (B win_h win_w) (h w) C",
                win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    return x


def window_reverse(windows, window_size_h, window_size_w, H, W, mode="basic"):
    if mode == "basic":  # [L B' C]
        B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
        x = rearrange(windows, "(h w) (B win_h win_w) C -> B C (win_h h) (win_w w)",
                B=B, win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    elif mode == "poolformer":  # [B' C H' W']
        B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
        x = rearrange(windows, "(B win_h win_w) C h w -> B C (win_h h) (win_w w)",
                B=B, win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    elif mode == "batch_first":  # [B' L C]
        B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
        x = rearrange(windows, "(B win_h win_w) (h w) C -> B C (win_h h) (win_w w)",
                B=B, win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    return x  # [B C H W]


def window_reverse_output(windows, window_size_h, window_size_w, H, W, mode="basic"):
    if mode == "basic":  # [L B' C]
        B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
        x = rearrange(windows, "(h w) (B win_h win_w) C -> (win_h h win_w w) B C",
                B=B, win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    elif mode == "poolformer":  # [B' C H' W']
        B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
        x = rearrange(windows, "(B win_h win_w) C h w -> (win_h h win_w w) B C",
                B=B, win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    elif mode == "batch_first":  # [B' L C]
        B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
        x = rearrange(windows, "(B win_h win_w) (h w) C -> (win_h h win_w w) B C",
                B=B, win_h=H//window_size_h, h=window_size_h, win_w=W//window_size_w, w=window_size_w)
    return x  # [(HxW), B, C]
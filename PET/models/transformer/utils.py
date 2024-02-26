"""
Transformer window-rize tools
"""

def enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w):
    """
    window-rize input for encoder
    """
    src_win = window_partition(src, window_size_h=enc_win_h, window_size_w=enc_win_w)  # [512, 16, 256]
    pos_embed_win = window_partition(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w)
    mask_win = window_partition(mask.unsqueeze(1), window_size_h=enc_win_h, window_size_w=enc_win_w).squeeze(-1).permute(1,0)
    return src_win, pos_embed_win, mask_win


def enc_win_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input for encoder
    """
    # windows.shape = [512, 16, 256]
    # 16 / (32*32/16/32) = 16(배치 크기 x 이미지 당 윈도우 개수) / 2(이미지 당 윈도우 개수) = 8(배치 크기)
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    # windows.shape = [512, 16, 256]
    # windows.permute = [16, 512, 256]
    # windows.view = [8, 2, 1, 16, 32, 256]
    x = windows.permute(1,0,2).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    # windows.permute = [8, 2, 16, 1, 32, 256]
    # windows.reshape = [8, 32, 32, 256]
    # windows.permute = [8, 256, 32, 32]
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0,3,1,2)
    return x  # [8, 256, 32, 32]


def window_partition(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    # [8, 256, 32, 32]
    B, C, H, W = x.shape
    # to (B, H, W, C) = (8, 32, 32, 256)
    x = x.permute(0,2,3,1)
    
    # [8, 2, 16, 1, 32, 256] = 이미지의 세로를 2등분, 가로를 1등분 = 윈도우 2개
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    
    # x.permute = [8, 2, 1, 16, 32, 256] = [배치 크기, 세로 조각 개수, 가로 조각 개수, 윈도우 세로 크기, 윈도우 가로 크기, 채널 수]
    # x.reshape = [16, 16, 32, 256] = [배치 크기 x 이미지 당 윈도우 개수, 윈도우 세로 크기, 윈도우 가로 크기, 채널 수]
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C)
    
    # windows.reshape = [16, 512, 256] = [배치 크기 x 이미지 당 윈도우 개수, 윈도우 벡터화, 채널 수]
    # windows.permute = [512, 16, 256] = [윈도우 벡터화, 배치 크기 x 이미지 당 윈도우 개수, 채널 수]
    windows = windows.reshape(-1, window_size_h*window_size_w, C).permute(1,0,2)
    return windows  # [512, 16, 256] = [윈도우 벡터화, 시퀀스 길이(배치 크기 x 이미지 당 윈도우 개수), 엠베딩]


def window_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.reshape(B, H*W, -1).permute(1,0,2)
    return x


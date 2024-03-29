"""
Transformer window-rize tools
"""
def enc_win_partition_p(src, enc_win_h, enc_win_w):
    """
    window-rize input for encoder
    """
    src_win = window_partition_p(src, window_size_h=enc_win_h, window_size_w=enc_win_w)  # [16, 256, 16, 32]
    return src_win


def enc_win_partition_reverse_p(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input for encoder
    """
    # windows.shape =  [16, 256, 16, 32]
    # 16 / (32*32/16/32) = 16(배치 크기 x 이미지 당 윈도우 개수) / 2(이미지 당 윈도우 개수) = 8(배치 크기)
    B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
    # windows.shape = [16, 256, 16, 32]
    # windows.view = [8, 2, 1, 256, 16, 32]
    x = windows.view(B, H // window_size_h, W // window_size_w, -1 ,window_size_h, window_size_w)
    # windows.permute = [8, 256, 2, 16, 1, 32]
    # windows.reshape = [8, 256, 32, 32]
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, -1, H, W)
    return x  # [8, 256, 32, 32]


def window_partition_p(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    # [8, 256, 32, 32]
    B, C, H, W = x.shape
    
    # # to (B, H, W, C) = (8, 32, 32, 256)
    # x = x.permute(0,2,3,1)
    
    # [8, 256, 2, 16, 1, 32] = 이미지의 세로를 2등분, 가로를 1등분 = 윈도우 2개
    x = x.reshape(B, C, H // window_size_h, window_size_h, W // window_size_w, window_size_w)
    
    # x.permute = [8, 2, 1, 256, 16, 32] = [배치 크기, 세로 조각 개수, 가로 조각 개수, 채널 수, 윈도우 세로 크기, 윈도우 가로 크기]
    # x.reshape = [16, 256, 16, 32] = [배치 크기 x 이미지 당 윈도우 개수, 윈도우 세로 크기, 윈도우 가로 크기, 채널 수]
    windows = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, window_size_h, window_size_w)
    
    return windows  # [16, 256, 16, 32] = [배치 크기 x 이미지 당 윈도우 개수, 윈도우 세로 크기, 윈도우 가로 크기, 채널 수]

def window_partition_L(x, window_size_h, window_size_w):
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

'''
def window_partition_reverse_p(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    # windows.shape =  [16, 256, 16, 32]
    # 16 / (32*32/16/32) = 16(배치 크기 x 이미지 당 윈도우 개수) / 2(이미지 당 윈도우 개수) = 8(배치 크기)
    B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
    # windows.reshape = [8, 2, 1, 256, 16, 32]
    x = windows.reshape(B, H // window_size_h, W // window_size_w, -1, window_size_h, window_size_w)
    # x.permute = [8, 256, 2, 16, 1, 32]
    # x.reshape = [8, 256, 1024]
    # x.permute = [1024, 8, 256]
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, -1, H*W).permute(2,0,1)
    return x
'''

def window_partition_reverse_p(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    # windows.shape =  [16, 256, 16, 32]
    B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
    x = windows.permute(0, 2, 3, 1).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.reshape(B, H*W, -1).permute(1,0,2)
    return x



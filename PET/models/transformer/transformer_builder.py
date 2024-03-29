def build_encoder(args, **kwargs):
    # kwargs : enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]
    # d_model=256, dropout=0.0, nhead=8, dim_feedforward=512,num_encoder_layers=4
    if args.transformer_method == 'swin':
        from .prog_swin_transformer import WinEncoderTransformer
        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            activation="gelu",  #### CHANGE
            **kwargs,
        )
    elif args.transformer_method == 'swinpool':
        from .prog_swin_poolformer import WinEncoderTransformer
        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            activation="gelu",  #### CHANGE
            **kwargs,
        )
    elif args.transformer_method == 'two_encoder':
        from .prog_2_encoder_win_transformer import WinEncoderTransformer
        kwargs['enc_win_list'] = [(32, 16), (16, 8)]
        kwargs['transformer_method'] = args.transformer_method
        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=len(kwargs['enc_win_list']),
            activation="gelu",  #### CHANGE
            **kwargs,
        )
    elif args.transformer_method == 'two_encoder_pooling':
        from .prog_2_encoder_win_transformer_pooling_NEW import WinEncoderTransformer
        kwargs['enc_win_list'] = [(32, 16), (16, 8)]
        kwargs['transformer_method'] = args.transformer_method
        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=len(kwargs['enc_win_list']),
            activation="gelu",  #### CHANGE
            **kwargs,
        )
    elif args.transformer_method == 'three_encoder':
        from .prog_2_encoder_win_transformer import WinEncoderTransformer
        kwargs['enc_win_list'] = [(32, 16), (32, 16), (16, 8)]
        kwargs['transformer_method'] = args.transformer_method
        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=len(kwargs['enc_win_list']),
            activation="gelu",  #### CHANGE
            **kwargs,
        )
    elif args.transformer_method == 'swin_patch_merging_FPN64_2Encoder':
        from .prog_2_encoder_swin_transformer import WinEncoderTransformer, PatchMerging
        
        kwargs['enc_win_list'] =[(32, 16), (32, 16)]

        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=len(kwargs['enc_win_list']),
            activation="gelu",  #### CHANGE
            downsample=PatchMerging,
            **kwargs,
        )
    elif args.transformer_method == 'HiLo':
        from .prog_win_transformer_HiLo_encoder import WinEncoderTransformer
        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            activation="gelu",  #### CHANGE
            **kwargs,
        )
    else: # if Encoder is Basic
        from .prog_win_transformer import WinEncoderTransformer
        return WinEncoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            activation="gelu",  #### CHANGE
            **kwargs,
        )


def build_decoder(args, **kwargs):
    if args.transformer_method == 'swin':
        from .prog_swin_transformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )
    elif args.transformer_method == 'swinpool':
        from .prog_swin_poolformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )
    elif args.transformer_method == 'two_encoder':
        from .prog_2_encoder_win_transformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )
    elif args.transformer_method == 'two_encoder_pooling':
        from .prog_2_encoder_win_transformer_pooling_NEW import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )
    elif args.transformer_method == 'swin_patch_merging_FPN64_2Encoder':
        from .prog_2_encoder_swin_transformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )
    else: # if Decoder is Basic
        from .prog_win_transformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            # return_intermediate_dec=True,
        )

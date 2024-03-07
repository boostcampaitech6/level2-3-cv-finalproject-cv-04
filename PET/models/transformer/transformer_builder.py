def build_encoder(args, **kwargs):
    # kwargs : enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]
    # d_model=256, dropout=0.0, nhead=8, dim_feedforward=512,num_encoder_layers=4
    if args.swin:
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
    elif args.swinpool:
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
    else:
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
    if args.swin:
        from .prog_swin_transformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )
    elif args.swinpool:
        from .prog_swin_poolformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )
    else:
        from .prog_win_transformer import WinDecoderTransformer
        return WinDecoderTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_decoder_layers=args.dec_layers,
            return_intermediate_dec=True,
        )

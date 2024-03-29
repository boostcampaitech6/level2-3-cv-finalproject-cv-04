def build_model(args):
    if args.pet_method == "basic":
        from .pet import build_pet
        return build_pet(args)
    elif args.pet_method == "all_pooling":
        from .pet_pooling import build_pet
        return build_pet(args)
    elif args.pet_method == "only_encoder_v1":
        from .pet_only_encoder_v1 import build_pet
        return build_pet(args)
    elif args.pet_method == "one_encoder":
        from .pet_one_encoder_v1 import build_pet
        return build_pet(args)
    elif args.pet_method == "only_encoder_v2":
        from .pet_only_encoder_v2 import build_pet
        return build_pet(args)
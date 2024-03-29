def build_model(args):
    if args.pet_method == "basic":
        from .pet import build_pet
        return build_pet(args)
    elif args.pet_method == "only_encoder":
        from .pet_only_encoder import build_pet
        return build_pet(args)
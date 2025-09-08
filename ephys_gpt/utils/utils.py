def key_args_str(_key_args: dict[str, str]) -> str:
    """Convert key args to a single string."""
    key_args = [f"{k}-{v}" for k, v in _key_args.items()]
    key_args = [arg.replace("_", "") for arg in key_args]
    return "_".join(key_args)

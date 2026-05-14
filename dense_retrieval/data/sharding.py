def get_shard_bounds(
    num_items: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """
    Compute the half-open interval [start, end) owned by one MPI rank.

    The first ranks receive one extra item if num_items is not divisible
    by world_size.

    Example:
        num_items = 10, world_size = 3

        rank 0 -> [0, 4)
        rank 1 -> [4, 7)
        rank 2 -> [7, 10)
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}")

    base = num_items // world_size
    remainder = num_items % world_size

    if rank < remainder:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        end = start + base

    return start, end

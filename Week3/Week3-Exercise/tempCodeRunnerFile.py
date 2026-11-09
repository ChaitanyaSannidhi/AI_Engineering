

    elif dataset_type == "borderline":

        return any(
            allocations[s] == capacities[s]
            for s in capacities
        ) or total_allocation == demand

    return False
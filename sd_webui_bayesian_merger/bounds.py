import warnings
from typing import Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf

from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS


class Bounds:
    @staticmethod
    def set_block_bounds(
        block_name: str, lb: float = 0.0, ub: float = 1.0
    ) -> Tuple[float, float]:
        return (lb, ub)

    @staticmethod
    def default_bounds(
        greek_letters: List[str],
        custom_ranges: Dict[str, Tuple[float, float]] = None,
    ) -> Dict:
        block_names = []
        ranges = {}
        for greek_letter in greek_letters:
            block_names += [
                f"block_{i}_{greek_letter}" for i in range(NUM_TOTAL_BLOCKS)
            ] + [f"base_{greek_letter}"]
            ranges |= {b: (0.0, 1.0) for b in block_names} | OmegaConf.to_object(
                custom_ranges
            )

        return {b: Bounds.set_block_bounds(b, *ranges[b]) for b in block_names}

    @staticmethod
    def freeze_bounds(bounds: Dict, frozen: Dict[str, float] = None) -> Dict:
        return {b: r for b, r in bounds.items() if b not in frozen}

    @staticmethod
    def group_bounds(bounds: Dict, groups: List[List[str]] = None) -> Dict:
        for group in groups:
            if not group:
                continue
            ranges = {bounds[b] for b in group if b in bounds}
            if len(ranges) > 1:
                w = (
                    f"different values for the same group: {group}"
                    + f" we're picking {group[0]} range: {bounds[group[0]]}"
                )
                warnings.warn(w)
                group_range = bounds[group[0]]
            elif ranges:
                group_range = ranges.pop()
            else:
                # all frozen
                continue

            group_name = "-".join(group)
            bounds[group_name] = group_range
            for b in group:
                if b in bounds:
                    del bounds[b]
        return bounds

    @staticmethod
    def get_bounds(
        greek_letters: List[str],
        frozen_params: Dict[str, float] = None,
        custom_ranges: Dict[str, Tuple[float, float]] = None,
        groups: List[List[str]] = None,
    ) -> Dict:
        if frozen_params is None:
            frozen_params = {}
        if custom_ranges is None:
            custom_ranges = DictConfig({})
        if groups is None:
            groups = []

        bounds = Bounds.default_bounds(greek_letters, custom_ranges)
        not_frozen_bounds = Bounds.freeze_bounds(bounds, frozen_params)
        grouped_bounds = Bounds.group_bounds(not_frozen_bounds, groups)
        return Bounds.freeze_groups(grouped_bounds, groups, frozen_params)

    @staticmethod
    def freeze_groups(bounds, groups, frozen):
        for group in groups:
            group_name = "-".join(group)
            if group_name in bounds and any(b in frozen for b in group):
                del bounds[group_name]
        return bounds

    @staticmethod
    def get_value(params, block_name, frozen, groups) -> float:
        if block_name in params:
            return params[block_name]
        if groups is not None:
            for group in groups:
                if block_name in group:
                    group_name = "-".join(group)
                    if group_name in params:
                        return params[group_name]
                    if group[0] in frozen:
                        return frozen[group[0]]
                    if group[0] in params:
                        return params[group[0]]
        return frozen[block_name]

    @staticmethod
    def assemble_params(
        params: Dict,
        greek_letters: List[str],
        frozen: Dict,
        groups: List[List[str]],
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        if frozen is None:
            frozen = {}
        if groups is None:
            groups = []

        weights = {}
        bases = {}

        for greek_letter in greek_letters:
            w = []
            for i in range(NUM_TOTAL_BLOCKS):
                block_name = f"block_{i}_{greek_letter}"
                value = Bounds.get_value(params, block_name, frozen, groups)
                w.append(value)

            assert len(w) == NUM_TOTAL_BLOCKS
            weights[greek_letter] = w

            base_name = f"base_{greek_letter}"
            bases[greek_letter] = Bounds.get_value(params, base_name, frozen, groups)

        assert len(weights) == len(greek_letters)
        assert len(bases) == len(greek_letters)
        return weights, bases

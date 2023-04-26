import pytest
from omegaconf import DictConfig, ListConfig

from sd_webui_bayesian_merger.bounds import Bounds
from sd_webui_bayesian_merger.merger import NUM_TOTAL_BLOCKS


def test_set_block_bounds():
    result = Bounds.set_block_bounds("block_1_alpha")
    assert result == (0.0, 1.0)

    result = Bounds.set_block_bounds("block_1_alpha", 3.0, 4.0)
    assert result == (3.0, 4.0)


def test_default_bounds():
    greek_letters = ["alpha", "beta"]
    custom_ranges = DictConfig({})
    bound = (0.0, 1.0)
    result = Bounds.default_bounds(greek_letters, custom_ranges)
    expected = (
        {f"block_{i}_alpha": bound for i in range(NUM_TOTAL_BLOCKS)}
        | {"base_alpha": bound}
        | {f"block_{i}_beta": bound for i in range(NUM_TOTAL_BLOCKS)}
        | {"base_beta": bound}
    )

    greek_letters = ["alpha"]
    custom_ranges = DictConfig({"block_1_alpha": (0.2, 0.4), "base_alpha": (0.2, 0.4)})
    result = Bounds.default_bounds(greek_letters, custom_ranges)
    expected = (
        {f"block_{i}_alpha": bound for i in range(NUM_TOTAL_BLOCKS) if i != 1}
        | {"base_alpha": (0.2, 0.4)}
        | {"block_1_alpha": (0.2, 0.4)}
    )
    assert result == expected


def test_freeze_bounds():
    greek_letters = ["alpha"]
    frozen = {"block_1_alpha": 0.1, "base_alpha": 0.3}
    custom_ranges = DictConfig({})
    bounds = Bounds.default_bounds(greek_letters, custom_ranges)
    result = Bounds.freeze_bounds(bounds, frozen)
    expected = {
        f"block_{i}_alpha": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS) if i != 1
    }
    assert result == expected


def test_group_bounds():
    greek_letters = ["alpha"]
    custom_ranges = DictConfig({})
    bounds = Bounds.default_bounds(greek_letters, custom_ranges)
    groups = [["block_0_alpha", "block_1_alpha"]]
    result = Bounds.group_bounds(bounds, groups)
    expected = (
        {f"block_{i}_alpha": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS) if i > 1}
        | {"base_alpha": (0.0, 1.0)}
        | {"block_0_alpha-block_1_alpha": (0.0, 1.0)}
    )
    assert result == expected
    assert len(result) == NUM_TOTAL_BLOCKS + 1 - 1

    custom_ranges = DictConfig({"base_alpha": (0.2, 0.3)})
    bounds = Bounds.default_bounds(greek_letters, custom_ranges)
    non_homogeneous_groups = [["base_alpha", "block_0_alpha"]]
    with pytest.warns(Warning):
        result = Bounds.group_bounds(bounds, non_homogeneous_groups)
    expected = {
        f"block_{i}_alpha": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS) if i != 0
    } | {"base_alpha-block_0_alpha": (0.2, 0.3)}
    assert result == expected
    assert len(result) == NUM_TOTAL_BLOCKS + 1 - 1

    frozen = {"block_1_alpha": 0.1, "base_alpha": 0.3, "block_0_alpha": 0.5}
    custom_ranges = DictConfig({})
    bounds = Bounds.default_bounds(greek_letters, custom_ranges)
    unfrozen_bounds = Bounds.freeze_bounds(bounds, frozen)
    frozen_groups = [["block_0_alpha", "block_1_alpha"]]
    result = Bounds.group_bounds(unfrozen_bounds, frozen_groups)
    expected = {
        f"block_{i}_alpha": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS) if i > 1
    }
    assert result == expected
    assert len(result) == NUM_TOTAL_BLOCKS + 1 - 3

    frozen = {"block_1_alpha": 0.1, "base_alpha": 0.3, "block_0_alpha": 0.5}
    custom_ranges = DictConfig({})
    bounds = Bounds.default_bounds(greek_letters, custom_ranges)
    unfrozen_bounds = Bounds.freeze_bounds(bounds, frozen)
    frozen_groups = [["block_0_alpha", "block_1_alpha", "base_alpha"]]
    result = Bounds.group_bounds(unfrozen_bounds, frozen_groups)
    expected = {
        f"block_{i}_alpha": (0.0, 1.0) for i in range(NUM_TOTAL_BLOCKS) if i > 1
    }
    assert result == expected
    assert len(result) == NUM_TOTAL_BLOCKS + 1 - 3


def test_assemble_params():
    greek_letters = ["alpha"]
    frozen = DictConfig({})
    custom_ranges = DictConfig({})
    groups = ListConfig([[]])

    bounds = Bounds.get_bounds(greek_letters, frozen, custom_ranges, groups)
    params = {b: r[0] for b, r in bounds.items()}
    weights, bases = Bounds.assemble_params(params, greek_letters, frozen, groups)

    assert all(w == 0.0 for w in weights["alpha"])
    assert bases["alpha"] == 0.0

    frozen = DictConfig({"block_1_alpha": 0.1, "base_alpha": 0.3, "block_0_alpha": 0.5})
    bounds = Bounds.get_bounds(greek_letters, frozen, custom_ranges, groups)
    params = {b: r[0] for b, r in bounds.items()}
    weights, bases = Bounds.assemble_params(params, greek_letters, frozen, groups)

    assert weights["alpha"][0] == 0.5
    assert weights["alpha"][1] == 0.1
    assert bases["alpha"] == 0.3

    bounds = Bounds.get_bounds(greek_letters, frozen, custom_ranges, groups)
    params = {b: r[0] for b, r in bounds.items()}
    weights, bases = Bounds.assemble_params(params, greek_letters, frozen, groups)

    frozen = DictConfig({"block_1_alpha": 0.1, "block_0_alpha": 0.5})
    non_homogeneous_groups = ListConfig([["block_1_alpha", "block_0_alpha"]])
    bounds = Bounds.get_bounds(
        greek_letters, frozen, custom_ranges, non_homogeneous_groups
    )
    params = {b: r[0] for b, r in bounds.items()}
    weights, bases = Bounds.assemble_params(
        params, greek_letters, frozen, non_homogeneous_groups
    )
    assert weights["alpha"][0] == 0.1
    assert weights["alpha"][1] == 0.1
    assert bases["alpha"] == 0.0

    frozen = DictConfig({"block_1_alpha": 0.1, "block_0_alpha": 0.5})
    non_homogeneous_groups = ListConfig(
        [["block_1_alpha", "block_0_alpha", "base_alpha"]]
    )
    bounds = Bounds.get_bounds(
        greek_letters, frozen, custom_ranges, non_homogeneous_groups
    )
    params = {b: r[0] for b, r in bounds.items()}
    weights, bases = Bounds.assemble_params(
        params, greek_letters, frozen, non_homogeneous_groups
    )
    assert weights["alpha"][0] == 0.1
    assert weights["alpha"][1] == 0.1
    assert bases["alpha"] == 0.1

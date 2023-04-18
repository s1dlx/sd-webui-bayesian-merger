import pytest
from omegaconf import DictConfig, ListConfig

from sd_webui_bayesian_merger.optimiser import BoundsInitialiser


def test_get_block_bounds():
    optimiser = "bayes"
    result = BoundsInitialiser.get_block_bounds(optimiser, "block_1_alpha")
    assert result == (0.0, 1.0)

    result = BoundsInitialiser.get_block_bounds(optimiser, "block_1_alpha", 3.0, 4.0)
    assert result == (3.0, 4.0)


def test_get_greek_letter_bounds():
    optimiser = "bayes"
    greek_letter = "alpha"
    custom_ranges = DictConfig({})
    result = BoundsInitialiser.get_greek_letter_bounds(
        greek_letter, optimiser, custom_ranges=custom_ranges
    )
    assert len(result) == 26
    assert all(greek_letter in k for k in result)
    assert all(b == (0.0, 1.0) for b in result.values())
    expected = {f"block_{i}_alpha": (0.0, 1.0) for i in range(25)} | {
        "base_alpha": (0.0, 1.0)
    }
    assert result == expected


def test_get_bounds():
    optimiser = "bayes"
    greek_letters = ["alpha", "beta"]
    result = BoundsInitialiser.get_bounds(greek_letters, optimiser)
    expected = (
        {f"block_{i}_alpha": (0.0, 1.0) for i in range(25)}
        | {"base_alpha": (0.0, 1.0)}
        | (
            {f"block_{i}_beta": (0.0, 1.0) for i in range(25)}
            | {"base_beta": (0.0, 1.0)}
        )
    )
    assert result == expected


def test_frozen_params():
    optimiser = "bayes"
    greek_letters = ["alpha"]
    frozen_params = DictConfig({"block_1_alpha": 0.1})
    result = BoundsInitialiser.get_bounds(
        greek_letters, optimiser, frozen_params=frozen_params
    )
    expected = {f"block_{i}_alpha": (0.0, 1.0) for i in range(25) if i != 1} | {
        "base_alpha": (0.0, 1.0)
    }
    assert result == expected


def test_custom_ranges():
    optimiser = "bayes"
    greek_letters = ["alpha"]
    custom_ranges = DictConfig({"block_1_alpha": (0.1, 0.2)})
    result = BoundsInitialiser.get_bounds(
        greek_letters, optimiser, custom_ranges=custom_ranges
    )
    expected = {
        f"block_{i}_alpha": (0.1, 0.2) if i == 1 else (0.0, 1.0) for i in range(25)
    } | {"base_alpha": (0.0, 1.0)}
    assert result == expected


def test_groups():
    optimiser = "bayes"
    greek_letters = ["alpha"]
    custom_ranges = DictConfig({})
    groups = ListConfig([['block_1_alpha', 'block_2_alpha']])
    result = BoundsInitialiser.get_bounds(
        greek_letters, optimiser, custom_ranges=custom_ranges, groups=groups,
    )
    expected = {
        f"block_{i}_alpha": (0.0, 1.0) for i in range(25) if i not in [1,2]
    } | {"base_alpha": (0.0, 1.0)} |{'block_1_alpha-block_2_alpha': (0.0, 1.0)}
    assert result == expected
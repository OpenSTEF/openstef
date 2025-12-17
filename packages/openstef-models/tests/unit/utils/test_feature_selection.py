# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the FeatureSelection utility."""

from openstef_models.utils.feature_selection import (
    Exclude,
    ExcludeRegex,
    FeatureSelection,
    Include,
    IncludeRegex,
)


def test_feature_selection_all():
    """Test FeatureSelection.ALL selects all features."""
    features = ["a", "b", "c"]
    assert FeatureSelection.ALL.resolve(features) == features


def test_feature_selection_none():
    """Test FeatureSelection.NONE selects no features."""
    features = ["a", "b", "c"]
    assert FeatureSelection.NONE.resolve(features) == []


def test_feature_selection_include():
    """Test including specific features."""
    selection = Include("a", "c")
    assert selection.resolve(["a", "b", "c", "d"]) == ["a", "c"]


def test_feature_selection_exclude():
    """Test excluding specific features."""
    selection = Exclude("b", "d")
    assert selection.resolve(["a", "b", "c", "d"]) == ["a", "c"]


def test_feature_selection_include_and_exclude():
    """Test combination of include and exclude."""
    selection = FeatureSelection(include={"a", "b", "c"}, exclude={"b"})
    assert selection.resolve(["a", "b", "c", "d"]) == ["a", "c"]


def test_feature_selection_combine_both_none():
    """Test combining two ALL selections preserves None."""
    combined = FeatureSelection.ALL.combine(FeatureSelection.ALL)
    assert combined.include is None
    assert combined.exclude is None
    assert combined.resolve(["a", "b", "c"]) == ["a", "b", "c"]


def test_feature_selection_combine_include_sets():
    """Test combining include sets."""
    sel1 = Include("a", "b")
    sel2 = Include("c", "d")
    combined = sel1.combine(sel2)
    assert set(combined.resolve(["a", "b", "c", "d", "e"])) == {"a", "b", "c", "d"}


def test_feature_selection_combine_mixed():
    """Test combining selections with different patterns."""
    sel1 = FeatureSelection(include={"a", "b"}, exclude={"b"})
    sel2 = FeatureSelection(include={"c"}, exclude={"a"})
    combined = sel1.combine(sel2)
    assert combined.include == {"a", "b", "c"}
    assert combined.exclude == {"a", "b"}
    assert combined.resolve(["a", "b", "c", "d"]) == ["c"]  # exclusion applied last


def test_regex_include_pattern():
    """Test including features by regex pattern."""
    selection = IncludeRegex(r"^temp_.*")
    features = ["temp_sensor", "temp_valve", "pressure_sensor", "humidity"]
    assert selection.resolve(features) == ["temp_sensor", "temp_valve"]


def test_regex_exclude_pattern():
    """Test excluding features by regex pattern."""
    selection = ExcludeRegex(r".*_old$")
    features = ["temp_new", "pressure_old", "humidity_current", "wind_old"]
    assert selection.resolve(features) == ["temp_new", "humidity_current"]


def test_regex_include_and_exclude():
    """Test combination of include and exclude regex patterns."""
    selection = FeatureSelection(include_regex={r"^temp_.*", r"^pressure_.*"}, exclude_regex={r".*_old$"})
    features = ["temp_sensor", "temp_old", "pressure_valve", "humidity_sensor", "pressure_old"]
    assert selection.resolve(features) == ["temp_sensor", "pressure_valve"]


def test_exact_and_regex_include():
    """Test combining exact and regex include patterns."""
    selection = FeatureSelection(include={"a"}, include_regex={r"^b.*"})
    features = ["a", "b1", "b2", "c"]
    assert set(selection.resolve(features)) == {"a", "b1", "b2"}


def test_exact_and_regex_exclude():
    """Test combining exact and regex exclude patterns."""
    selection = FeatureSelection(exclude={"a"}, exclude_regex={r"^b.*"})
    features = ["a", "b1", "b2", "c"]
    assert selection.resolve(features) == ["c"]


def test_combine_exact_and_regex():
    """Test combining selections with exact and regex patterns."""
    sel1 = Include("a", "b")
    sel2 = IncludeRegex(r"^c.*")
    combined = sel1.combine(sel2)
    features = ["a", "b", "c1", "c2", "d"]
    assert set(combined.resolve(features)) == {"a", "b", "c1", "c2"}


def test_combine_all_types():
    """Test combining all types of patterns."""
    sel1 = FeatureSelection(include={"a"}, exclude_regex={r".*_old$"})
    sel2 = FeatureSelection(include_regex={r"^temp_.*"}, exclude={"a"})
    combined = sel1.combine(sel2)
    features = ["a", "temp_sensor", "temp_old", "pressure"]
    # include: {a} + regex temp_.*
    # exclude: {a} + regex .*_old$
    # So: a excluded, temp_sensor included, temp_old excluded
    assert combined.resolve(features) == ["temp_sensor"]

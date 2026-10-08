"""Exact synthetic inventory, independent of image extraction and reporting I/O."""
import json
from pathlib import Path

from datasets.synthetic import DEFAULT_MANIFEST
from experiments.protocol import control_combinations, primary_combinations
from photometric import PHOTOMETRIC_VALUES, synthetic_grid


def validate_synthetic_inventory(frame, auxiliary=False):
    combinations = set(control_combinations() if auxiliary else primary_combinations())
    expected_count = 8 * 40 * len(combinations)
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} synthetic {'auxiliary' if auxiliary else 'primary'} records")
    frozen = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["sources"]
    sources = set(frame.source_image)
    if len(sources) != 8 or {Path(p).parent.name + "/" + Path(p).name for p in sources} != set(frozen):
        raise ValueError("Synthetic sources differ from the frozen eight-image manifest")
    settings = synthetic_grid()
    expected = {(method, strategy, source, s.kind, s.parameter)
                for method, strategy in combinations for source in sources for s in settings}
    columns = ["method", "matching_strategy", "source_image", "transformation_type", "transformation_strength"]
    if frame.duplicated(columns).any() or set(frame[columns].itertuples(index=False, name=None)) != expected:
        raise ValueError("Missing/duplicate/mislabeled synthetic combination, source or level")
    if set(frame.dataset) != {"synthetic"} or not frame.is_primary.eq(not auxiliary).all():
        raise ValueError("Synthetic primary/auxiliary labels are inconsistent")
    names = {(s.kind, s.parameter): s.name for s in settings}
    for row in frame.itertuples():
        sequence = Path(row.source_image).parent.name
        pair = sequence + "__" + names[(row.transformation_type, row.transformation_strength)]
        family = "photometric" if row.transformation_type in PHOTOMETRIC_VALUES else "geometry"
        if (row.sequence != sequence or row.pair != pair or row.synthetic_pair_id != pair
                or row.transformation_family != family or row.category != row.transformation_type
                or row.identity_baseline_id != sequence + "__identity_0"):
            raise ValueError("Inconsistent synthetic pair identity/family/baseline")

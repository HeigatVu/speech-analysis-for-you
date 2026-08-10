"""Optional standardized acoustic feature packs."""

from .definitions import EGEMAPS_KEYS, RAW_EGEMAPS_COLUMNS, register_egemaps_features
from .opensmile_adapter import extract_egemaps_features

register_egemaps_features()

__all__ = [
    "EGEMAPS_KEYS",
    "RAW_EGEMAPS_COLUMNS",
    "extract_egemaps_features",
]

"""Stable definitions for the annotation-driven ``motor_neuro`` pack."""

from __future__ import annotations

from ...catalog import FeatureDefinition, register_feature

ARTICULATION_KEYS = (
    "artic_vowel_space_area_hz2",
    "artic_vowel_articulation_index",
    "artic_formant_centralization_ratio",
    "artic_vowel_dispersion_mean_hz",
    "artic_vowel_dispersion_sd_hz",
    "artic_f1_within_vowel_sd_hz",
    "artic_f2_within_vowel_sd_hz",
    "artic_formant_transition_slope_mean_hz_s",
    "artic_vot_mean_s",
    "artic_vot_sd_s",
    "artic_stop_gap_mean_s",
    "artic_stop_gap_sd_s",
    "artic_consonant_duration_mean_s",
    "artic_consonant_duration_sd_s",
    "artic_fricative_m1_mean",
    "artic_fricative_m2_mean",
    "artic_fricative_m3_mean",
    "artic_fricative_m4_mean",
    "artic_resonance_attenuation_mean_db",
)

RHYTHM_KEYS = (
    "rhythm_percent_vocalic",
    "rhythm_varco_v",
    "rhythm_varco_c",
    "rhythm_rpvi_v",
    "rhythm_npvi_v",
    "rhythm_rpvi_c",
    "rhythm_npvi_c",
)

DDK_KEYS = (
    "task_ddk_rate_syllables_s",
    "task_ddk_inter_onset_mean_s",
    "task_ddk_inter_onset_median_s",
    "task_ddk_inter_onset_sd_s",
    "task_ddk_inter_onset_cv",
    "task_ddk_instability_s",
    "task_ddk_acceleration_syllables_s2",
    "task_ddk_decay_ratio",
    "task_ddk_voiced_interval_mean_s",
    "task_ddk_sequential_alternating_ratio",
)

RESPIRATORY_KEYS = (
    "resp_breath_group_count",
    "resp_breath_group_mean_s",
    "resp_breath_group_sd_s",
    "resp_rate_per_min",
    "resp_pauses_per_breath",
    "resp_relative_loudness_db",
)

SUSTAINED_VOWEL_KEYS = (
    "task_max_phonation_time_s",
    "voice_gaping_interval_rate_per_min",
    "voice_subharmonic_interval_proportion",
    "voice_sustained_f0_sd_semitones",
    "voice_sustained_power_sd_db",
)

ALL_KEYS = ARTICULATION_KEYS + RHYTHM_KEYS + DDK_KEYS + RESPIRATORY_KEYS + SUSTAINED_VOWEL_KEYS

_UNITS = {
    "artic_vowel_space_area_hz2": "Hz2",
    "artic_vowel_articulation_index": "ratio",
    "artic_formant_centralization_ratio": "ratio",
    "artic_vowel_dispersion_mean_hz": "Hz",
    "artic_vowel_dispersion_sd_hz": "Hz",
    "artic_f1_within_vowel_sd_hz": "Hz",
    "artic_f2_within_vowel_sd_hz": "Hz",
    "artic_formant_transition_slope_mean_hz_s": "Hz/s",
    "artic_vot_mean_s": "s",
    "artic_vot_sd_s": "s",
    "artic_stop_gap_mean_s": "s",
    "artic_stop_gap_sd_s": "s",
    "artic_consonant_duration_mean_s": "s",
    "artic_consonant_duration_sd_s": "s",
    "artic_fricative_m1_mean": "coefficient",
    "artic_fricative_m2_mean": "coefficient",
    "artic_fricative_m3_mean": "coefficient",
    "artic_fricative_m4_mean": "coefficient",
    "artic_resonance_attenuation_mean_db": "dB",
    "rhythm_percent_vocalic": "percent",
    "rhythm_varco_v": "percent",
    "rhythm_varco_c": "percent",
    "rhythm_rpvi_v": "s",
    "rhythm_npvi_v": "percent",
    "rhythm_rpvi_c": "s",
    "rhythm_npvi_c": "percent",
    "task_ddk_rate_syllables_s": "syllables/s",
    "task_ddk_inter_onset_mean_s": "s",
    "task_ddk_inter_onset_median_s": "s",
    "task_ddk_inter_onset_sd_s": "s",
    "task_ddk_inter_onset_cv": "ratio",
    "task_ddk_instability_s": "s",
    "task_ddk_acceleration_syllables_s2": "syllables/s2",
    "task_ddk_decay_ratio": "ratio",
    "task_ddk_voiced_interval_mean_s": "s",
    "task_ddk_sequential_alternating_ratio": "ratio",
    "task_max_phonation_time_s": "s",
    "resp_breath_group_count": "count",
    "resp_breath_group_mean_s": "s",
    "resp_breath_group_sd_s": "s",
    "resp_rate_per_min": "count/min",
    "resp_pauses_per_breath": "ratio",
    "resp_relative_loudness_db": "dB",
    "voice_gaping_interval_rate_per_min": "count/min",
    "voice_subharmonic_interval_proportion": "ratio",
    "voice_sustained_f0_sd_semitones": "semitones",
    "voice_sustained_power_sd_db": "dB",
}


def _domain(key: str) -> str:
    if key.startswith("artic_"):
        return "articulation"
    if key.startswith("rhythm_"):
        return "rhythm"
    if key.startswith("resp_"):
        return "respiration"
    if key.startswith("task_"):
        return "task"
    return "phonation"


def _tasks(key: str) -> tuple[str, ...]:
    if key.startswith("task_ddk_"):
        return ("ddk",)
    if key in SUSTAINED_VOWEL_KEYS:
        return ("sustained_vowel",)
    return ("connected_speech",)


_DEFINITIONS = tuple(
    FeatureDefinition(
        key=key,
        pack="motor_neuro",
        level="recording",
        unit=_UNITS[key],
        population="adult",
        reference="Neurodegenerative speech feature expansion v1",
        domain=_domain(key),
        language_scope="language_sensitive"
        if key.startswith(("artic_", "rhythm_"))
        else "language_independent",
        tasks=_tasks(key),
        disorders=("als", "mnd", "pd", "pdd", "hd", "ms", "ataxia", "psp", "msa", "cbs"),
        evidence_level="derived_companion",
    )
    for key in ALL_KEYS
)

_registered = False


def register_motor_features() -> None:
    """Register the motor pack once."""
    global _registered
    if _registered:
        return
    for definition in _DEFINITIONS:
        register_feature(definition)
    _registered = True


__all__ = [
    "ALL_KEYS",
    "ARTICULATION_KEYS",
    "DDK_KEYS",
    "RESPIRATORY_KEYS",
    "RHYTHM_KEYS",
    "SUSTAINED_VOWEL_KEYS",
    "register_motor_features",
]

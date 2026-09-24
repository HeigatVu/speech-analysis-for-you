# Adult-Neuro Linguistic Feature Pack (`src/speech_features/features/linguistic/`)

The **`adult_neuro`** pack provides **183 registered features** tailored for cognitive-linguistic characterization, neurodegenerative language breakdown (Alzheimer's Disease, Primary Progressive Aphasia, Frontotemporal Dementia, Mild Cognitive Impairment), and structured discourse analysis.

All features are registered in [`definitions.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/linguistic/definitions.py) under pack name `"adult_neuro"`.

---

## 1. Module Inventory

| Module | Features | Description | Key Indicators |
|---|---|---|---|
| [`definitions.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/linguistic/definitions.py) | — | Registration schema, units, levels, prerequisites, and formula versions | Metadata definitions for all 183 features |
| [`diversity.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/linguistic/diversity.py) | 16 | Lexical diversity measures computed independently on surface tokens and lemmatized tokens | `lex_token_ttr`, `lex_token_mattr_20`, `lex_token_mtld`, `lex_token_hdd_42`, `lex_token_brunet_w`, `lex_token_honore_r`, `lex_token_entropy`, plus corresponding `lex_lemma_*` keys |
| [`clinical.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/linguistic/clinical.py) | 22 | Speech disfluencies, speech errors, and repair mechanisms | `disfluency_filler_count`, `disfluency_filler_ratio`, `disfluency_fragment_count`, `disfluency_immediate_repetition_count`, `disfluency_retracing_count`, `disfluency_revision_count` |
| [`morphosyntax.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/linguistic/morphosyntax.py) | 38 | Universal Dependencies (UD-VTB) part-of-speech distributions and syntactic complexity | `pos_noun_ratio`, `pos_verb_ratio`, `pos_noun_to_verb_ratio`, `pos_open_to_closed_ratio`, `syn_mean_dependency_length`, `syn_tree_depth_mean`, `syn_subordinate_clause_ratio` |
| [`task_scores.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/linguistic/task_scores.py) | 107 | Discourse coherence, information efficiency, examiner prompts, and picture description task scoring | `discourse_information_unit_count`, `discourse_information_efficiency_per_min`, `discourse_causal_cohesion_ratio`, `discourse_referential_cohesion_ratio`, `discourse_examiner_prompt_ratio`, `discourse_response_latency_mean_s`, `concept_cat_mentioned`, `action_running_mentioned` |

---

## 2. Vietnamese Language Integrity

1. **Surface Compound Word Preservation:**
   Vietnamese multi-syllable compound words (e.g., `bệnh_nhân`, `bác_sĩ`) grouped by Underthesea maintain NFC normalization, exact tone marks, and `d`/`đ`. Syllable counts and word lengths are calculated accurately without stripping compound boundaries.
2. **Pretokenized Morphosyntactic Parsing:**
   Stanza parses pretokenized compound words with Multi-Word Token (MWT) expansion disabled, preserving compound lemmas in the `%mor` tier (e.g., `noun|bệnh_nhân`).
3. **Discourse & Task Specs:**
   Cognitive tasks (such as Cookie Theft or picture descriptions) use structured task specifications (`task_spec`) defining concept aliases, action groups, and entity categories.

---

## 3. Usage

```python
import speech_features as sf

document = sf.load_document("session_001.cha")
task_spec = {
    "version": 1,
    "task": "cookie_theft",
    "concept_aliases": {"boy": ["cậu_bé", "bé_trai"], "cookie": ["bánh_quy", "bánh"]},
    "entity_groups": {},
    "action_groups": {},
}

bundle = sf.extract(
    "session_001.wav",
    document,
    packs=("adult_neuro",),
    task_spec=task_spec,
    target_speakers={"PAR"},
)

print(bundle.recordings[["lex_token_ttr", "lex_lemma_mtld", "discourse_information_unit_count"]])
```

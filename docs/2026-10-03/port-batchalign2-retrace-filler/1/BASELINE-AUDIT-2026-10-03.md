# Pinned baseline audit

Source was read directly during planning on 2026-10-03. Historical T1/T3/T4 permission blockers are settled by this source evidence and Chatter 0.28.0 validation, independently of agent summaries.

Supporting files: [Batchalign2 strip](https://github.com/TalkBank/batchalign2/blob/d8bb0cd05d968b31333e6cfbbd4e93b93d288236/batchalign/document.py#L333), [UD morphotagger](https://github.com/TalkBank/batchalign2/blob/d8bb0cd05d968b31333e6cfbbd4e93b93d288236/batchalign/pipelines/morphosyntax/ud.py#L750), [Batchalign3 features](https://github.com/FranklinChen/talkbank-tools/blob/f5f4235255a473042272d55a0581fcd58152334f/crates/batchalign-transform/src/morphosyntax/features.rs), [retrace transform](https://github.com/FranklinChen/talkbank-tools/blob/f5f4235255a473042272d55a0581fcd58152334f/crates/batchalign-transform/src/asr_postprocess/retrace.rs), [Chatter alignment](https://github.com/TalkBank/chatter/blob/5c414cf65fec64358dcec804ed777646eb8b35f4/book/src/architecture/alignment.md), [wor timing](https://github.com/TalkBank/chatter/blob/5c414cf65fec64358dcec804ed777646eb8b35f4/book/src/architecture/wor-timing.md), [Sherpa VAD](https://github.com/welcomyou/sherpa-vietnamese-asr/blob/350f5a4c569d714cfe574e36e9a2913b81912ba1/core/vad_utils.py), and [ASR engine](https://github.com/welcomyou/sherpa-vietnamese-asr/blob/350f5a4c569d714cfe574e36e9a2913b81912ba1/core/asr_engine.py).

| Source | Commit | Evidence |
|---|---|---|
| [FranklinChen/talkbank-tools](https://github.com/FranklinChen/talkbank-tools/tree/f5f4235255a473042272d55a0581fcd58152334f) | f5f4235255a473042272d55a0581fcd58152334f | Batchalign3 transform morphology and ASR retrace source |
| [TalkBank/batchalign](https://github.com/TalkBank/batchalign/tree/ec35582dd29bca959e61eae1a0e44787f6ca4dc3) | ec35582dd29bca959e61eae1a0e44787f6ca4dc3 | Rust PCM preparation and Chatter contracts |
| [TalkBank/chatter](https://github.com/TalkBank/chatter/tree/5c414cf65fec64358dcec804ed777646eb8b35f4) | 5c414cf65fec64358dcec804ed777646eb8b35f4 | Alignment and wor timing architecture |
| [TalkBank/batchalign2](https://github.com/TalkBank/batchalign2/tree/d8bb0cd05d968b31333e6cfbbd4e93b93d288236) | d8bb0cd05d968b31333e6cfbbd4e93b93d288236 | document.py strip defaults and morphotagger call |
| [Sherpa Vietnamese ASR](https://github.com/welcomyou/sherpa-vietnamese-asr/tree/350f5a4c569d714cfe574e36e9a2913b81912ba1) | 350f5a4c569d714cfe574e36e9a2913b81912ba1 | VAD copy gain/retry, silence chunking and disabled repetition cleanup |

## Verified rules

Batchalign2 document.py `strip(join_with_spaces=False, include_retrace=False, include_fp=False)` includes REGULAR/PUNCT by default; RETRACE/FP require explicit inclusion. The UD morphotagger calls strip with joined spaces, without either opt-in. This directly establishes the former unverified %mor exclusion rule.

Chatter mor/gra skip fillers, retraces, fragments, nonwords, untranscribed speech and pauses; replacement annotations supply the corrected morphology words. Wor describes original spoken words and retains fillers/retraces. Equal item counts alone cannot corroborate lexical timing: edited main-tier words must not inherit stale wor timing. Nonpositive spans cannot be admitted.

Batchalign3 feature mapping is POS-specific, ordered and not arbitrary UD-value serialization. Verbs order VerbForm/Aspect/Mood/Tense/Polarity/Polite followed by compact number-person. Pronouns order PronType/Case/reflexive/number-person. Determiners and nouns use different gender/number suppression rules. Function words carry no suffixes. Vietnamese surface words remain unchanged.

## Intentional divergences

- Both upstream audio helpers include mean-downmix paths. The repository's declared-channel rule is stricter and wins; the upstream citation does not establish a never-downmix guarantee. Native PCM stays intact and model resampling is in memory.
- Upstream retrace marking supports longer repeated spans and protects split seams. The local heuristic is capped at four and respects punctuation boundaries. It remains provisional for human review; a repeated clinical word is not automatically a hallucination.
- Sherpa VAD peaks a detector copy at 0.071 only for quiet input. Its public VAD helper defaults to 0.2, internal inference defaults to 0.5, and empty retry explicitly uses 0.3/150 ms. Local retry halves the configured threshold; copying 0.3 after 0.2 would be stricter. Historical T7 text is superseded by this correction.
- Sherpa silence cuts use a 300 ms quiet run near a midpoint, 30 s chunks and 3 s overlap. Local 20 s cuts search the preceding 2 s and cover audio without overlap. This is an adaptation, not numerical equivalence.
- Sherpa's repeated n-gram deletion is disabled to preserve legitimate Vietnamese repeats. Do not import content deletion, backchannel removal, global ASR gain or new models without separate evidence.

## Defects assigned to follow-up tasks

Approved hash comparison before decode (T9); canonical suffix mapping, replacement/retrace scope and lexical timing admission (T10); surgical tag output with MED/task preservation (T11); spoken-domain evaluation (T12). Pilot accuracy/performance claims require new measured T13 evidence, not the historical September aggregate.

## Validator evidence

The local Chatter binary is version 0.28.0. Its extracted bytes match the release archive; the archive matches the published SHA-256 checksum. Eleven synthetic focused validator cases passed during planning without skips. Cargo/rustc installation is unnecessary. Re-run after corrective implementation with CHATTER_BIN explicitly set.

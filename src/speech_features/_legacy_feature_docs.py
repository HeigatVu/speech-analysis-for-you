"""Feature-specific formula and missing-data text preserved from catalog v1."""

LEGACY_FEATURE_DOCS = {
    "audio_clipping_ratio": (
        "Fraction of samples with absolute amplitude at or beyond the source width clipping boundary 1 - 2^-(bits-1) (32767/32768 for 16-bit).",
        "Always defined for decodable PCM audio; undecodable audio raises INVALID_AUDIO before extraction.",
    ),
    "audio_dc_offset": (
        "Sample mean of the mono float signal (normalized amplitude; 0 for digital silence).",
        "Always defined for decodable PCM audio; undecodable audio raises INVALID_AUDIO before extraction.",
    ),
    "audio_duration_s": (
        "Total recording length: len(audio) / sample_rate in seconds.",
        "Always defined for decodable PCM audio; undecodable audio raises INVALID_AUDIO before extraction.",
    ),
    "audio_rms_dbfs": (
        "20*log10(rms) with rms = sqrt(mean(x^2)); digital silence (rms == 0) is NaN, never -inf.",
        "NaN plus a NO_AUDIO warning issue when the recording is digital silence.",
    ),
    "discourse_examiner_prompt_ratio": (
        "Prompted target turns divided by target turns; a target turn is prompted when the immediately preceding ordered turn belongs to a non-target speaker.",
        "NaN plus MISSING_ANNOTATION when the document has no non-target turns.",
    ),
    "discourse_overlap_ratio": (
        "discourse_overlap_s divided by the union duration of target intervals; prerequisite discourse_overlap_s.",
        "NaN plus INSUFFICIENT_TOKENS with zero target utterance duration; MISSING_ANNOTATION when the document has no non-target turns.",
    ),
    "discourse_overlap_s": (
        "Union duration of all target/non-target interval intersections at recording level.",
        "NaN plus MISSING_ANNOTATION when the document has no non-target turns.",
    ),
    "discourse_response_latency_mean_s": (
        "Mean of the defined per-turn response latencies.",
        "NaN plus MISSING_ANNOTATION with no defined latencies or no non-target turns.",
    ),
    "discourse_response_latency_s": (
        "max(0, target.start_s minus previous non-target end_s) for a prompted target turn.",
        "NaN plus MISSING_ANNOTATION when the turn is not prompted or no non-target predecessor exists.",
    ),
    "discourse_response_latency_sd_s": (
        "Population SD of the defined per-turn response latencies; requires at least two.",
        "NaN plus INSUFFICIENT_TOKENS with fewer than two defined latencies; MISSING_ANNOTATION with no defined latencies or no non-target turns.",
    ),
    "discourse_turn_count": (
        "Number of explicit target-speaker utterances (turns); the document is ordered by (start_s, end_s, id).",
        "NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "discourse_turn_length_mean_syllables": (
        "Total explicit syllables divided by the number of target turns.",
        "NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "discourse_turn_length_mean_words": (
        "Total explicit words divided by the number of target turns.",
        "NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "discourse_turn_length_sd_syllables": (
        "Population SD of per-turn syllable counts; requires at least two target turns.",
        "NaN plus INSUFFICIENT_TOKENS with fewer than two target turns; MISSING_ANNOTATION without a target language sample.",
    ),
    "discourse_turn_length_sd_words": (
        "Population SD of per-turn word counts; requires at least two target turns.",
        "NaN plus INSUFFICIENT_TOKENS with fewer than two target turns; MISSING_ANNOTATION without a target language sample.",
    ),
    "discourse_turn_overlap_s": (
        "Union duration of intersections of the target turn interval with all non-target intervals.",
        "NaN plus MISSING_ANNOTATION when the document has no non-target turns.",
    ),
    "discourse_turn_syllable_count": (
        "Explicit syllables of one target turn (word-kind tokens).",
        "NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "discourse_turn_word_count": (
        "Explicit words of one target turn (same word grouping as lex_word_count).",
        "NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_annotated_error_count": (
        "Count of explicit target tokens with kind error.",
        "0 for a present target sample with no such event; NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_annotated_error_ratio": (
        "Annotated-error count divided by lex_token_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero denominator; MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_filler_count": (
        "Count of explicit target tokens with kind filler.",
        "0 for a present target sample with no such event; NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_filler_ratio": (
        "Filler count divided by lex_token_count (all target tokens).",
        "NaN plus INSUFFICIENT_TOKENS with a zero denominator; MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_fragment_count": (
        "Count of explicit target tokens with kind fragment.",
        "0 for a present target sample with no such event; NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_fragment_ratio": (
        "Fragment count divided by lex_token_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero denominator; MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_immediate_repetition_count": (
        "Count of word-kind tokens whose NFC plus casefold text equals the immediately previous word-kind token in the same utterance; adjacent pairs never cross utterances.",
        "0 for a present target sample with no such event; NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_immediate_repetition_ratio": (
        "Immediate-repetition count divided by the explicit syllable count (lex_syllable_count).",
        "NaN plus INSUFFICIENT_TOKENS with a zero denominator; MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_maze_count": (
        "Sum of filler, fragment, immediate repetition, retracing, and revision counts; errors and noise are excluded.",
        "0 for a present target sample with no such event; NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_maze_ratio": (
        "Maze count divided by lex_token_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero denominator; MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_retracing_count": (
        "Count of explicit target tokens with kind retracing.",
        "0 for a present target sample with no such event; NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_retracing_ratio": (
        "Retracing count divided by lex_token_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero denominator; MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_revision_count": (
        "Count of explicit target tokens with kind revision.",
        "0 for a present target sample with no such event; NaN plus MISSING_ANNOTATION without a target language sample.",
    ),
    "disfluency_revision_ratio": (
        "Revision count divided by lex_token_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero denominator; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_character_count": (
        "Sum of NFC code points over the target's word-kind token texts.",
        "NaN plus MISSING_ANNOTATION when the target has no explicit language sample (no tokens); a present sample with no event of a kind is a real zero.",
    ),
    "lex_lemma_brunet_w": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): N to the power (V to the power -0.165).",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_lemma_entropy": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): Normalized Shannon entropy -sum(p*log(p))/log(V); a single-type non-empty sample is defined as 0.",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_lemma_hapax_ratio": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): V1 divided by N (hapax types over tokens).",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_lemma_hdd_42": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): HD-D with sample size 42: sum over types of (1 - C(N-f,42)/C(N,42)) divided by 42, computed as the bounded product product(k=0..41, (N-f-k)/(N-k)); NaN when N is below 42.",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_lemma_honore_r": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): 100*log(N) divided by (1 - V1/V); NaN when the divisor vanishes (V equals V1).",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_lemma_mattr_20": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): Mean TTR over every contiguous length-20 window; NaN when N is below 20.",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_lemma_mtld": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): Mean of the forward and reverse MTLD factor methods with threshold 0.72 and partial factor (1 - ttr)/(1 - 0.72); NaN when no factor ever closes.",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_lemma_ttr": (
        "Over the lemma annotation layer values (must cover every target word token; no surface fallback): V divided by N (types over tokens of the normalized sequence).",
        "NaN plus MISSING_ANNOTATION when the lemma layer is absent or incomplete for the target word tokens; INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_mlu_syllables": (
        "lex_syllable_count divided by lex_utterance_count; prerequisites lex_syllable_count and lex_utterance_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero utterance count; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_mlu_words": (
        "lex_word_count divided by lex_utterance_count; prerequisites lex_word_count and lex_utterance_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero utterance count; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_syllable_count": (
        "Number of explicit word-kind tokens; each is one orthographic syllable observation and text is never split on whitespace.",
        "NaN plus MISSING_ANNOTATION when the target has no explicit language sample (no tokens); a present sample with no event of a kind is a real zero.",
    ),
    "lex_token_brunet_w": (
        "Over the NFC plus casefold surface word-kind token sequence: N to the power (V to the power -0.165).",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_count": (
        "Count of every explicit target token, including CHAT code tokens of any kind.",
        "NaN plus MISSING_ANNOTATION when the target has no explicit language sample (no tokens); a present sample with no event of a kind is a real zero.",
    ),
    "lex_token_entropy": (
        "Over the NFC plus casefold surface word-kind token sequence: Normalized Shannon entropy -sum(p*log(p))/log(V); a single-type non-empty sample is defined as 0.",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_hapax_ratio": (
        "Over the NFC plus casefold surface word-kind token sequence: V1 divided by N (hapax types over tokens).",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_hdd_42": (
        "Over the NFC plus casefold surface word-kind token sequence: HD-D with sample size 42: sum over types of (1 - C(N-f,42)/C(N,42)) divided by 42, computed as the bounded product product(k=0..41, (N-f-k)/(N-k)); NaN when N is below 42.",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_honore_r": (
        "Over the NFC plus casefold surface word-kind token sequence: 100*log(N) divided by (1 - V1/V); NaN when the divisor vanishes (V equals V1).",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_length_mean_characters": (
        "Mean NFC code-point length of the target's word-kind token texts.",
        "NaN plus INSUFFICIENT_TOKENS with no word-kind tokens; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_length_sd_characters": (
        "Population SD of word-token lengths; requires at least two word tokens.",
        "NaN plus INSUFFICIENT_TOKENS with fewer than two word tokens; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_mattr_20": (
        "Over the NFC plus casefold surface word-kind token sequence: Mean TTR over every contiguous length-20 window; NaN when N is below 20.",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_mtld": (
        "Over the NFC plus casefold surface word-kind token sequence: Mean of the forward and reverse MTLD factor methods with threshold 0.72 and partial factor (1 - ttr)/(1 - 0.72); NaN when no factor ever closes.",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_token_ttr": (
        "Over the NFC plus casefold surface word-kind token sequence: V divided by N (types over tokens of the normalized sequence).",
        "NaN plus INSUFFICIENT_TOKENS when the sample is too small or degenerate; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_unique_token_count": (
        "Number of distinct NFC plus casefold word-kind token texts; diacritics and d versus đ are preserved in the text and normalized only for comparison.",
        "NaN plus MISSING_ANNOTATION when the target has no explicit language sample (no tokens); a present sample with no event of a kind is a real zero.",
    ),
    "lex_utterance_count": (
        "Count of explicit target-speaker DocumentUtterance rows.",
        "NaN plus MISSING_ANNOTATION when the target has no explicit language sample (no tokens); a present sample with no event of a kind is a real zero.",
    ),
    "lex_word_count": (
        "Number of explicit words: word-kind tokens grouped by non-null word_id within each utterance; an ungrouped word token is one word; grouping never crosses utterances.",
        "NaN plus MISSING_ANNOTATION when the target has no explicit language sample (no tokens); a present sample with no event of a kind is a real zero.",
    ),
    "lex_word_length_mean_syllables": (
        "Mean syllables per explicit word (grouped by word_id within an utterance).",
        "NaN plus INSUFFICIENT_TOKENS with no explicit words; MISSING_ANNOTATION without a target language sample.",
    ),
    "lex_word_length_sd_syllables": (
        "Population SD of syllables per explicit word; requires at least two explicit words.",
        "NaN plus INSUFFICIENT_TOKENS with fewer than two explicit words; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_classifier_ratio": (
        "Mean of the classifier annotation layer over all target word tokens (numeric 0 or 1 per token).",
        "NaN plus MISSING_ANNOTATION when the classifier layer is absent, incomplete, or holds non-binary values.",
    ),
    "morph_clause_count": (
        "Count of clause-head tokens with relations root, ccomp, xcomp, advcl, acl, acl:relcl, csubj, or csubj:pass.",
        "0 for a present target sample with no clause heads; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_clause_rate_per_utterance": (
        "morph_clause_count divided by the target utterance count; prerequisites morph_clause_count and lex_utterance_count.",
        "NaN plus INSUFFICIENT_TOKENS with zero target utterances; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_code_switch_ratio": (
        "Share of target word tokens whose token.language (NFC plus casefold) differs from the document language.",
        "NaN plus MISSING_ANNOTATION when not every target word token carries a non-empty language.",
    ),
    "morph_content_word_ratio": (
        "(ADJ + ADV + NOUN + PROPN + VERB) divided by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_adverbial_modifier_ratio": (
        "Share of target word tokens whose casefolded relation is advmod, obl, obl:agent, or dislocated.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_clausal_complement_ratio": (
        "Share of target word tokens whose casefolded relation is ccomp, xcomp, advcl, acl, or acl:relcl.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_coordination_ratio": (
        "Share of target word tokens whose casefolded relation is conj or cc.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_function_ratio": (
        "Share of target word tokens whose casefolded relation is case, mark, det, aux, aux:pass, cop, or clf.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_nominal_modifier_ratio": (
        "Share of target word tokens whose casefolded relation is amod, nmod, nmod:poss, appos, or compound.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_object_ratio": (
        "Share of target word tokens whose casefolded relation is obj or iobj.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_other_ratio": (
        "Share of every remaining non-empty relation; the nine ratios partition the target word tokens and sum to one.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_root_ratio": (
        "Share of target word tokens whose casefolded relation is root.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dep_subject_ratio": (
        "Share of target word tokens whose casefolded relation is nsubj, nsubj:pass, csubj, or csubj:pass.",
        "NaN plus MISSING_ANNOTATION when any target word token lacks a relation, an utterance's word tokens do not have exactly one root, or a head chain fails to reach the root (cycle); MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_dependency_length_mean_tokens": (
        "Mean of abs(dependent position - head position) over non-root edges inside the same utterance; requires at least one edge.",
        "NaN plus INSUFFICIENT_TOKENS with no dependency edges; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_dependency_length_sd_tokens": (
        "Population SD of dependency lengths; requires at least two edges.",
        "NaN plus INSUFFICIENT_TOKENS with fewer than two dependency edges; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_function_word_ratio": (
        "(ADP + AUX + CCONJ + DET + PART + PRON + SCONJ) divided by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_noun_verb_ratio": (
        "(NOUN + PROPN) divided by VERB.",
        "NaN plus INSUFFICIENT_TOKENS with a zero verb denominator; MISSING_ANNOTATION when the upos layer is unusable or there is no target language sample.",
    ),
    "morph_particle_ratio": (
        "Mean of the particle annotation layer over all target word tokens (numeric 0 or 1 per token).",
        "NaN plus MISSING_ANNOTATION when the particle layer is absent, incomplete, or holds non-binary values.",
    ),
    "morph_pronoun_noun_ratio": (
        "PRON divided by (NOUN + PROPN).",
        "NaN plus INSUFFICIENT_TOKENS with a zero noun plus proper-noun denominator; MISSING_ANNOTATION when the upos layer is unusable or there is no target language sample.",
    ),
    "morph_subordinate_clause_count": (
        "Count of clause heads excluding root.",
        "0 for a present target sample with no subordinate clauses; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_subordination_ratio": (
        "morph_subordinate_clause_count divided by morph_clause_count; prerequisites morph_subordinate_clause_count and morph_clause_count.",
        "NaN plus INSUFFICIENT_TOKENS with a zero clause count; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_tree_depth_max": (
        "Maximum tree depth over target word tokens.",
        "NaN plus INSUFFICIENT_TOKENS with no word-kind tokens; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_tree_depth_mean": (
        "Mean number of head edges from each target word token to its utterance root (root depth is 0).",
        "NaN plus INSUFFICIENT_TOKENS with no word-kind tokens; MISSING_ANNOTATION when the dependency analysis is unusable or there is no target language sample.",
    ),
    "morph_upos_adj_ratio": (
        "Share of target word-kind tokens tagged ADJ in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_adp_ratio": (
        "Share of target word-kind tokens tagged ADP in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_adv_ratio": (
        "Share of target word-kind tokens tagged ADV in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_aux_ratio": (
        "Share of target word-kind tokens tagged AUX in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_cconj_ratio": (
        "Share of target word-kind tokens tagged CCONJ in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_det_ratio": (
        "Share of target word-kind tokens tagged DET in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_intj_ratio": (
        "Share of target word-kind tokens tagged INTJ in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_noun_ratio": (
        "Share of target word-kind tokens tagged NOUN in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_num_ratio": (
        "Share of target word-kind tokens tagged NUM in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_part_ratio": (
        "Share of target word-kind tokens tagged PART in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_pron_ratio": (
        "Share of target word-kind tokens tagged PRON in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_propn_ratio": (
        "Share of target word-kind tokens tagged PROPN in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_punct_ratio": (
        "Share of target word-kind tokens tagged PUNCT in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_sconj_ratio": (
        "Share of target word-kind tokens tagged SCONJ in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_sym_ratio": (
        "Share of target word-kind tokens tagged SYM in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_verb_ratio": (
        "Share of target word-kind tokens tagged VERB in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "morph_upos_x_ratio": (
        "Share of target word-kind tokens tagged X in the upos annotation layer; ratios divide by all target word tokens.",
        "NaN plus MISSING_ANNOTATION when the upos layer is absent, incomplete, non-string, or holds an unknown tag; MISSING_ANNOTATION without a target language sample.",
    ),
    "spectral_b1_mean_hz": (
        "Mean of B1 (bandwidth of F1) over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_b1_sd_hz": (
        "Population SD of B1 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_b2_mean_hz": (
        "Mean of B2 (bandwidth of F2) over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_b2_sd_hz": (
        "Population SD of B2 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_b3_mean_hz": (
        "Mean of B3 (bandwidth of F3) over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_b3_sd_hz": (
        "Population SD of B3 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_centroid_mean_hz": (
        "Power-weighted mean frequency sum(f*P)/sum(P)over the real-FFT bins (P = rfft(frame) squared) of VAD-voiced target frames; mean over frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_centroid_sd_hz": (
        "Population SD over frames of the per-frame centroid sum(f*P)/sum(P).",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_entropy_mean": (
        "Normalized Shannon entropy of the power distribution p = P/sum(P): -sum(p*log(p))/log(n_bins), bounded in [0, 1]; mean over frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_entropy_sd": (
        "Population SD over frames of the per-frame spectral entropy.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_f1_mean_hz": (
        "Mean of F1 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_f1_sd_hz": (
        "Population SD of F1 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_f2_mean_hz": (
        "Mean of F2 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_f2_sd_hz": (
        "Population SD of F2 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_f3_mean_hz": (
        "Mean of F3 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_f3_sd_hz": (
        "Population SD of F3 over accepted formant frames. LPC order 12 (default) on pre-emphasized (0.97) Hamming frames; stable roots (abs below 1) below Nyquist map to f = theta*sr/(2*pi) and bandwidth b = -sr*log(rho)/pi; accepted frames yield three candidates ordered F1 below F2 below F3 with bandwidths B1-B3.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with no stable voiced frame; NaN plus INSUFFICIENT_FORMANTS with fewer than two accepted formant frames.",
    ),
    "spectral_flatness_mean": (
        "Geometric over arithmetic mean power exp(mean(log(P + 1e-12)))/mean(P + 1e-12), bounded in [0, 1]; mean over frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_flatness_sd": (
        "Population SD over frames of the per-frame flatness.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_flux_mean": (
        "L2 norm of the difference between consecutive unit-L2-normalized power spectra; pairs are formed only inside each target interval; mean over pairs.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_flux_sd": (
        "Population SD over pairs of the per-pair spectral flux.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_rolloff_85_mean_hz": (
        "Frequency of the first bin whose cumulative power reaches 85 percent of the frame total; mean over frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_rolloff_85_sd_hz": (
        "Population SD over frames of the per-frame rolloff frequency.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_slope_mean_db_per_hz": (
        "OLS slope of the dB power spectrum 10*log10(P) (floored at 1e-12) over frequency in Hz; mean over frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_slope_sd_db_per_hz": (
        "Population SD over frames of the per-frame spectral slope.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_spread_mean_hz": (
        "Power-weighted RMS deviation sqrt(sum(P*(f - centroid)^2)/sum(P))over the real-FFT bins (P = rfft(frame) squared) of VAD-voiced target frames; mean over frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "spectral_spread_sd_hz": (
        "Population SD over frames of the per-frame spread.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES with fewer than two voiced frames; flux keys additionally NaN plus INSUFFICIENT_SPEECH_FRAMES when no interval holds two consecutive voiced frames.",
    ),
    "time_articulation_rate_syllables_per_s": (
        "Syllables (word-kind tokens) divided by time_speech_s; prerequisite time_speech_s.",
        "NaN plus MISSING_ANNOTATION when the target has no word-kind tokens or no aligned target intervals.",
    ),
    "time_long_pause_count": (
        "Number of pauses of at least long_pause_threshold_s (2.0 s).",
        "0 when no pause meets 2.0 s; NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_overlap_s": (
        "Total seconds of intersection between merged target intervals and merged non-target intervals.",
        "NaN plus MISSING_ANNOTATION when the document has no non-target utterances.",
    ),
    "time_pause_count": (
        "Number of unvoiced runs of at least pause_threshold_s (0.2 s) inside the target intervals.",
        "NaN plus NO_SPEECH when no voiced frame exists; 0 when the target speaks but no pause meets pause_threshold_s.",
    ),
    "time_pause_max_s": (
        "Maximum qualifying pause duration.",
        "NaN when no pause meets pause_threshold_s; NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_pause_mean_s": (
        "Mean duration of qualifying pauses (at least pause_threshold_s).",
        "NaN when no pause meets pause_threshold_s; NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_pause_rate_per_min": (
        "time_pause_count divided by the recording duration in minutes.",
        "NaN plus NO_SPEECH when no voiced frame exists; 0 when no pause meets pause_threshold_s.",
    ),
    "time_pause_sd_s": (
        "Population SD of qualifying pause durations.",
        "NaN when no pause meets pause_threshold_s; NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_response_latency_s": (
        "Mean of utterance.start_s minus previous.end_s over target utterances whose immediately preceding utterance (latest end_s not above start_s) belongs to a non-target speaker.",
        "NaN plus MISSING_ANNOTATION when no examiner-to-participant response pair exists.",
    ),
    "time_speech_ratio": (
        "time_speech_s divided by the recording duration; prerequisite time_speech_s.",
        "NaN plus MISSING_ANNOTATION when time_speech_s is unavailable (no aligned target intervals without allow_unaligned).",
    ),
    "time_speech_s": (
        "Total duration of the target speaker's merged, recording-clipped aligned utterance intervals.",
        "NaN plus MISSING_ANNOTATION when no aligned target intervals exist (without allow_unaligned); the whole-recording fallback with allow_unaligned=True emits an UNALIGNED_SPEAKER warning.",
    ),
    "time_syllable_duration_cv": (
        "SD divided by the mean of the same token durations; requires at least two observations and a positive mean.",
        "NaN plus MISSING_ANNOTATION with no document, no word-kind timed tokens, or too few observations.",
    ),
    "time_syllable_duration_mean_s": (
        "Mean of (end_s minus start_s) over the target speaker's word-kind tokens with finite times and end_s above start_s; one observation per token, tokens sharing a word_id are not collapsed; requires at least one observation.",
        "NaN plus MISSING_ANNOTATION with no document, no word-kind timed tokens, or too few observations.",
    ),
    "time_syllable_duration_npvi": (
        "Normalized pairwise variability index 100 * mean(abs(d[i+1] - d[i]) / ((d[i+1] + d[i]) / 2)); pairs only between consecutive tokens inside the same utterance, pooled across the target's utterances; requires at least one intra-utterance pair.",
        "NaN plus MISSING_ANNOTATION with no document, no word-kind timed tokens, or too few observations.",
    ),
    "time_syllable_duration_sd_s": (
        "Population SD of the same token durations; requires at least two observations.",
        "NaN plus MISSING_ANNOTATION with no document, no word-kind timed tokens, or too few observations.",
    ),
    "time_syllables_per_min": (
        "Word-kind token count (orthographic syllables) divided by speech minutes; prerequisite time_speech_s.",
        "NaN plus MISSING_ANNOTATION when the target has no word-kind tokens or no aligned target intervals.",
    ),
    "time_voiced_segment_mean_s": (
        "Mean duration of voiced runs over VAD-voiced frames (25 ms frames, 10 ms hop, mean-energy VAD), pooled over target intervals.",
        "NaN plus NO_SPEECH when no voiced frame exists in the analyzed audio; MISSING_ANNOTATION when no aligned target intervals exist (without allow_unaligned).",
    ),
    "time_voiced_segment_sd_s": (
        "Population SD of voiced-run durations, pooled over target intervals.",
        "NaN plus NO_SPEECH when no voiced frame exists in the analyzed audio; MISSING_ANNOTATION when no aligned target intervals exist (without allow_unaligned).",
    ),
    "time_words_per_min": (
        "Explicit word count divided by speech minutes; words group by word_id within an utterance and ungrouped word tokens count one word each; prerequisite time_speech_s.",
        "NaN plus MISSING_ANNOTATION when the target has no word-kind tokens or no aligned target intervals.",
    ),
    "voice_cpp_iqr_db": (
        "Interquartile range of per-frame real-cepstrum peak prominence: the peak of the cepstrum of the dB log-magnitude spectrum over the pitch-period quefrency range, measured relative to its local OLS baseline and scaled by 20/ln(10).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_cpp_mean_db": (
        "Mean of per-frame real-cepstrum peak prominence: the peak of the cepstrum of the dB log-magnitude spectrum over the pitch-period quefrency range, measured relative to its local OLS baseline and scaled by 20/ln(10).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_cpp_median_db": (
        "Median of per-frame real-cepstrum peak prominence: the peak of the cepstrum of the dB log-magnitude spectrum over the pitch-period quefrency range, measured relative to its local OLS baseline and scaled by 20/ln(10).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_cpp_sd_db": (
        "Population SD of per-frame real-cepstrum peak prominence: the peak of the cepstrum of the dB log-magnitude spectrum over the pitch-period quefrency range, measured relative to its local OLS baseline and scaled by 20/ln(10).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_abs_change_hz": (
        "Mean absolute difference of consecutive voiced F0 estimates; pairs are formed only inside each target interval.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES with fewer than two voiced frames; NaN plus INSUFFICIENT_CYCLES when no single interval holds two voiced frames.",
    ),
    "voice_f0_cv": (
        "SD divided by the mean of per-frame F0 over voiced frames (VAD-voiced frames with a finite autocorrelation peak inside the pitch range, 70-400 Hz default; 25 ms/10 ms Hamming frames).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_iqr_hz": (
        "Interquartile range (P75 minus P25) of per-frame F0 over voiced frames (VAD-voiced frames with a finite autocorrelation peak inside the pitch range, 70-400 Hz default; 25 ms/10 ms Hamming frames).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_mean_hz": (
        "Mean of per-frame F0 over voiced frames (VAD-voiced frames with a finite autocorrelation peak inside the pitch range, 70-400 Hz default; 25 ms/10 ms Hamming frames).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_median_hz": (
        "Median of per-frame F0 over voiced frames (VAD-voiced frames with a finite autocorrelation peak inside the pitch range, 70-400 Hz default; 25 ms/10 ms Hamming frames).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_range_5_95_hz": (
        "P95 minus P05 of per-frame F0 over voiced frames (VAD-voiced frames with a finite autocorrelation peak inside the pitch range, 70-400 Hz default; 25 ms/10 ms Hamming frames).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_sd_hz": (
        "Population SD of per-frame F0 over voiced frames (VAD-voiced frames with a finite autocorrelation peak inside the pitch range, 70-400 Hz default; 25 ms/10 ms Hamming frames).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_slope_hz_per_s": (
        "OLS slope of F0 over the voiced frames' actual times (frame i timed at start_s + i * hop / sample_rate).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_hnr_iqr_db": (
        "Interquartile range of per-frame HNR = 10*log10(r/(1-r)) in dB from the best autocorrelation peak r (ceiling r at 0.999).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_hnr_mean_db": (
        "Mean of per-frame HNR = 10*log10(r/(1-r)) in dB from the best autocorrelation peak r (ceiling r at 0.999).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_hnr_median_db": (
        "Median of per-frame HNR = 10*log10(r/(1-r)) in dB from the best autocorrelation peak r (ceiling r at 0.999).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_hnr_sd_db": (
        "Population SD of per-frame HNR = 10*log10(r/(1-r)) in dB from the best autocorrelation peak r (ceiling r at 0.999).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_intensity_iqr_db": (
        "Interquartile range of frame intensity over VAD-voiced target frames (20*log10(rms) per frame).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_intensity_mean_dbfs": (
        "Mean of frame intensity over VAD-voiced target frames (20*log10(rms) per frame).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_intensity_median_dbfs": (
        "Median of frame intensity over VAD-voiced target frames (20*log10(rms) per frame).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_intensity_sd_db": (
        "Population SD of frame intensity over VAD-voiced target frames (20*log10(rms) per frame).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_intensity_slope_db_per_s": (
        "OLS slope of frame intensity over the voiced frames' actual times.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_jitter_local": (
        "Mean absolute consecutive period difference divided by the mean period; period = 1/f0 per voiced frame and pairs are formed only inside each target interval.",
        "NaN plus INSUFFICIENT_CYCLES when no interval holds two consecutive voiced frames; INSUFFICIENT_VOICED_FRAMES with fewer than two voiced frames.",
    ),
    "voice_shimmer_local": (
        "Same ratio over consecutive voiced-frame RMS amplitudes; pairs are formed only inside each target interval.",
        "NaN plus INSUFFICIENT_CYCLES when no interval holds two consecutive voiced frames; INSUFFICIENT_VOICED_FRAMES with fewer than two voiced frames.",
    ),
    "voice_voiced_ratio": (
        "Voiced F0 frames divided by all analyzed target frames (including silence).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when no voiced frame exists; digital silence never yields a finite voice feature.",
    ),
}

__all__ = ["LEGACY_FEATURE_DOCS"]

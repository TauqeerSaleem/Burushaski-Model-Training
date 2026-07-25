import jiwer
from sacrebleu.metrics import BLEU, CHRF, TER


def _pct(value):
    return round(value * 100, 2)


def _safe_accuracy(error_rate):
    return round(max(0.0, 1.0 - error_rate) * 100, 2)


def exact_match_rate(hypotheses, references):
    if not references:
        return 0.0
    matches = sum(1 for hyp, ref in zip(hypotheses, references) if str(hyp).strip() == str(ref).strip())
    return round(matches / len(references) * 100, 2)


def empty_prediction_rate(hypotheses):
    if not hypotheses:
        return 0.0
    empty = sum(1 for hyp in hypotheses if not str(hyp).strip())
    return round(empty / len(hypotheses) * 100, 2)


def mean_word_count(texts):
    if not texts:
        return 0.0
    return round(sum(len(str(text).split()) for text in texts) / len(texts), 2)


def mean_length_ratio(hypotheses, references):
    ratios = []
    for hyp, ref in zip(hypotheses, references):
        ref_len = max(1, len(str(ref).split()))
        ratios.append(len(str(hyp).split()) / ref_len)
    return round(sum(ratios) / len(ratios), 3) if ratios else 0.0


def asr_scores(hypotheses, references, prefix=""):
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    return {
        f"{prefix}wer_%": _pct(wer),
        f"{prefix}cer_%": _pct(cer),
        f"{prefix}word_accuracy_%": _safe_accuracy(wer),
        f"{prefix}char_accuracy_%": _safe_accuracy(cer),
        f"{prefix}sentence_error_rate_%": round(100.0 - exact_match_rate(hypotheses, references), 2),
        f"{prefix}exact_match_%": exact_match_rate(hypotheses, references),
        f"{prefix}empty_prediction_%": empty_prediction_rate(hypotheses),
        f"{prefix}mean_ref_words": mean_word_count(references),
        f"{prefix}mean_hyp_words": mean_word_count(hypotheses),
    }


def mt_scores(hypotheses, references):
    return {
        "bleu": round(BLEU().corpus_score(hypotheses, [references]).score, 2),
        "chrf++": round(CHRF(word_order=2).corpus_score(hypotheses, [references]).score, 2),
        "ter": round(TER().corpus_score(hypotheses, [references]).score, 2),
        "exact_match_%": exact_match_rate(hypotheses, references),
        "empty_prediction_%": empty_prediction_rate(hypotheses),
        "mean_ref_words": mean_word_count(references),
        "mean_hyp_words": mean_word_count(hypotheses),
        "mean_length_ratio": mean_length_ratio(hypotheses, references),
    }

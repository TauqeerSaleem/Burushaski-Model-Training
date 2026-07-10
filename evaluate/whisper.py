import argparse
import string
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from dotenv import load_dotenv
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data.loader import load_dataset

load_dotenv()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def main(args):
    print("Loading test data...")
    test_hf = load_dataset(
        split="test",
        use_hf=args.use_hf,
        use_supabase=False,
        use_mdc=args.use_mdc,
    )
    print(f"Test examples: {len(test_hf)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()
    print(f"Model loaded from {args.model_path}")

    hypotheses, references, filenames, failed = [], [], [], []

    print("\nRunning inference...")
    for row in tqdm(test_hf, desc="Evaluating"):
        try:
            reference = row["english_translation"]
            if not reference:
                continue

            audio, _ = librosa.load(row["audio"], sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(inputs, max_length=225)

            hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            hypotheses.append(hypothesis)
            references.append(reference)
            filenames.append(row["id"])

        except Exception as e:
            print(f"  Failed: {row['id']} — {e}")
            failed.append(row["id"])

    print(f"\nProcessed: {len(hypotheses)} | Failed: {len(failed)}")

    hyp_norm = [normalize_text(h) for h in hypotheses]
    ref_norm = [normalize_text(r) for r in references]

    print("\nComputing metrics...")
    wer = jiwer.wer(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)
    bleu_result = BLEU().corpus_score(hyp_norm, [ref_norm])
    chrf_result = CHRF(word_order=2).corpus_score(hyp_norm, [ref_norm])

    word_accuracies = []
    for pred, ref in zip(hyp_norm, ref_norm):
        pred_words, ref_words = set(pred.split()), set(ref.split())
        if ref_words:
            word_accuracies.append(len(pred_words & ref_words) / len(ref_words))
    word_overlap = np.mean(word_accuracies) * 100 if word_accuracies else 0.0

    print("Computing BERTScore...")
    valid = [(h, r) for h, r in zip(hypotheses, references) if h.strip() and r.strip()]
    try:
        _, _, F1 = bert_score(
            [p[0] for p in valid],
            [p[1] for p in valid],
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        bertscore_f1 = F1.mean().item()
    except Exception as e:
        print(f"BERTScore failed: {e}")
        bertscore_f1 = 0.0

    bleu = bleu_result.score
    quality = "Poor" if bleu < 10 else "Fair" if bleu < 20 else "Good" if bleu < 30 else "Excellent"

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples:        {len(hypotheses)}")
    print(f"  BLEU:           {bleu:.2f}")
    print(f"  chrF++:         {chrf_result.score:.2f}")
    print(f"  BERTScore F1:   {bertscore_f1:.4f}")
    print(f"  WER:            {wer*100:.2f}%")
    print(f"  CER:            {cer*100:.2f}%")
    print(f"  Word Overlap:   {word_overlap:.2f}%")
    print(f"  Quality:        {quality}")
    print("=" * 50)

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "filename": filenames,
        "reference": references,
        "hypothesis": hypotheses,
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm,
        "wer_%": [round(jiwer.wer(r, h) * 100, 2) for r, h in zip(ref_norm, hyp_norm)],
        "cer_%": [round(jiwer.cer(r, h) * 100, 2) for r, h in zip(ref_norm, hyp_norm)],
    }).to_csv(results_dir / "evaluation_results.csv", index=False)

    pd.DataFrame([{
        "Total_Samples": len(hypotheses),
        "BLEU": round(bleu, 2),
        "chrF++": round(chrf_result.score, 2),
        "BERTScore_F1": round(bertscore_f1, 4),
        "WER_%": round(wer * 100, 2),
        "CER_%": round(cer * 100, 2),
        "Word_Overlap_%": round(word_overlap, 2),
        "Quality": quality,
    }]).to_csv(results_dir / "metrics_summary.csv", index=False)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs/whisper_final")
    parser.add_argument("--output-dir", type=str, default="outputs/whisper/results")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-mdc", action="store_true", default=False)
    args = parser.parse_args()
    main(args)

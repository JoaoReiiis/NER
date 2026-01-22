import os
import json
import pandas as pd
import glob
import random
import time
from tqdm import tqdm
from seqeval.metrics import classification_report

from config import PASTA_DATASET, TOTAL_FOLDS, LIMIT_SAMPLES, BENCHMARK_MODELS, BENCHMARK_SHOTS
from engine import DataLoader, LLMEngine


def format_examples_for_log(examples):
    if not examples: return "Nenhum (Zero-Shot)"
    log_str = ""
    for idx, (tokens, _) in enumerate(examples):
        text = " ".join(tokens)
        log_str += f"[Ex{idx+1}: {text[:40]}...] "
    return log_str

def consolidate_experiment_metrics(experiment_folder):
    all_files = glob.glob(os.path.join(experiment_folder, "Fold_*", "metrics.csv"))
    if not all_files: return None

    df_list = [pd.read_csv(f) for f in all_files]
    if not df_list: return None

    full_df = pd.concat(df_list)

    summary = full_df.groupby('label').agg({
        'precision': ['mean', 'std'], 'recall': ['mean', 'std'],
        'f1-score': ['mean', 'std'], 'support': 'sum'
    }).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    final_path = os.path.join(experiment_folder, "RELATORIO_CONSOLIDADO.csv")
    summary.to_csv(final_path, index=False)
    row = summary[summary['label'] == 'micro avg']
    if row.empty: row = summary[summary['label'] == 'weighted avg']

    if not row.empty:
        return row.iloc[0]['f1-score_mean']
    return 0.0


def run_single_experiment(loader, model_name, n_shots):
    llm = LLMEngine(model_name)
    mode_name = "ZeroShot" if n_shots == 0 else f"FewShot_{n_shots}"

    experiment_dir = os.path.join("Resultados_Benchmark", model_name, mode_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"\nINICIANDO: {model_name} | {mode_name}")

    has_data = False

    for fold in range(1, TOTAL_FOLDS + 1):
        test_sents, true_bio = loader.load_fold(fold)

        if not test_sents:
            continue

        has_data = True

        train_sents, train_bio = [], []
        if n_shots > 0:
            train_sents, train_bio = loader.load_training_context(fold, TOTAL_FOLDS)

        tags_validas = sorted(list(loader.all_tags))
        if LIMIT_SAMPLES:
            test_sents = test_sents[:LIMIT_SAMPLES]
            true_bio = true_bio[:LIMIT_SAMPLES]

        fold_dir = os.path.join(experiment_dir, f"Fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        pred_bio_list = []
        feedback_rows = []

        desc_bar = f"Fold {fold}/{TOTAL_FOLDS}"

        for i, tokens in tqdm(enumerate(test_sents), total=len(test_sents), desc=desc_bar, leave=False):
            examples = []
            if n_shots > 0 and train_sents:
                k = min(n_shots, len(train_sents))
                if k > 0:
                    indices = random.sample(range(len(train_sents)), k)
                    examples = [(train_sents[idx], train_bio[idx]) for idx in indices]

            pred_ents, pred_bio, is_hallucination = llm.predict(tokens, tags_validas, few_shot_examples=examples)
            pred_bio_list.append(pred_bio)

            true_ents = llm.bio_to_json(tokens, true_bio[i])
            json_true = json.dumps(sorted(true_ents, key=lambda x: x['text']), ensure_ascii=False)
            json_pred = json.dumps(sorted(pred_ents, key=lambda x: x['text']), ensure_ascii=False)
            status = "ALUCINACAO" if is_hallucination else ("PERFEITO" if json_true == json_pred else "DIVERGENTE")

            feedback_rows.append({
                "texto": " ".join(tokens),
                "reais": json_true,
                "previstos": json_pred,
                "status": status,
                "exemplos": format_examples_for_log(examples)
            })

        pd.DataFrame(feedback_rows).to_csv(os.path.join(fold_dir, "feedback.csv"), index=False)

        report = classification_report(true_bio, pred_bio_list, output_dict=True, zero_division=0)
        metrics_data = []
        for label, scores in report.items():
            if isinstance(scores, dict):
                metrics_data.append({
                    "label": label, "precision": scores['precision'],
                    "recall": scores['recall'], "f1-score": scores['f1-score'], "support": scores['support']
                })
        pd.DataFrame(metrics_data).to_csv(os.path.join(fold_dir, "metrics.csv"), index=False)

    if not has_data:
        print(f" Nenhum dado encontrado para {model_name}. Verifique o DataLoader.")
        return 0.0

    final_f1 = consolidate_experiment_metrics(experiment_dir)

    if final_f1 is None:
        final_f1 = 0.0

    print(f" Concluido: {model_name} ({mode_name}) -> F1 Medio: {final_f1:.4f}")
    return final_f1

def run_benchmark():
    loader = DataLoader(PASTA_DATASET)

    print("="*60)
    print("X - BENCHMARK - X")
    print(f"Modelos: {BENCHMARK_MODELS}")
    print(f"Shots: {BENCHMARK_SHOTS}")
    print("="*60)

    global_results = []
    start_total = time.time()

    for model in BENCHMARK_MODELS:
        for shots in BENCHMARK_SHOTS:

            start_exp = time.time()
            f1_score = run_single_experiment(loader, model, shots)
            duration = time.time() - start_exp

            global_results.append({
                "Modelo": model,
                "Shots": shots,
                "Modo": "ZeroShot" if shots == 0 else f"FewShot_{shots}",
                "F1_Score_Medio": f1_score,
                "Tempo_Segundos": round(duration, 2)
            })
            pd.DataFrame(global_results).to_csv("Resultados_Benchmark/PROGRESSO_PARCIAL.csv", index=False)

    print("\n" + "="*60)
    print("RANKING...")
    print("="*60)

    df_results = pd.DataFrame(global_results)
    df_results = df_results.sort_values(by="F1_Score_Medio", ascending=False)

    final_path = "Resultados_Benchmark/RANKING_FINAL_GERAL.csv"
    df_results.to_csv(final_path, index=False)

    print(df_results.to_string(index=False))
    print(f"\nResultados detalhados em: Resultados_Benchmark/")
    print(f"Tempo Total: {(time.time() - start_total)/60:.1f} minutos")

if __name__ == "__main__":
    run_benchmark()

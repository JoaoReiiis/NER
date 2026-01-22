import os
import json
import time
import pandas as pd
from tqdm import tqdm
from seqeval.metrics import classification_report
from gliner import GLiNER
import torch
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from config import PASTA_DATASET, TOTAL_FOLDS, LIMIT_SAMPLES
from engine import DataLoader

TAG_MAPPING = {
    "PRODUTO_MARCA": "marca cachaça rótulo nome bebida",
    "PRODUTOR_MESTRE": "produtor mestre alambiqueiro fundador criador",
    "ORIGEM_GEOGRAFICA": "cidade estado origem fazenda região terroir local",
    "FABRICANTE_EMPRESA": "empresa fabricante cooperativa destilaria engenho",

    "TEOR_ALCOOLICO": "graduação alcoólica teor vol porcentagem % abv",
    "TEMPO_ENVELHECIMENTO": "envelhecimento maturação descanso anos meses idade",
    "MADEIRA_BARRIL": "madeira barril carvalho amburana bálsamo jequitibá",
    "VOLUME_FRASCO": "volume capacidade garrafa ml litros dose",
    "VALOR_COMERCIAL": "preço valor custa reais R$ investimento",
    "CATEGORIA_QUALIDADE": "classificação ouro prata premium extra-premium reserva",

    "METODO_DESTILACAO": "destilação alambique coluna capelo bitola",
    "RECIPIENTE_ARMAZENAMENTO": "tonel dorna barril tanque inox",

    "SENSORIAL_OLFATO": "aroma cheiro nariz notas olfato perfume",
    "SENSORIAL_PALADAR": "sabor gosto boca paladar adocicado acidez picância",
    "SENSORIAL_VISUAL": "cor visual coloração límpida dourada brilhante lágrimas",
    "SENSORIAL_TEXTURA": "consistência textura corpo oleosidade aveludada",
    "SENSORIAL_FINALIZACAO": "retrogosto final acabamento persistência fundo",

    "PERIODO_TEMPO": "tempo duração período safra"
}

REVERSE_MAPPING = {v: k for k, v in TAG_MAPPING.items()}

GLINER_CONFIGS = [
    #sem tags
    {"model": "knowledgator/gliner-multitask-large-v0.5", "threshold": 0.3, "use_mapping": False},
    {"model": "knowledgator/gliner-multitask-large-v0.5", "threshold": 0.5, "use_mapping": False},
    {"model": "knowledgator/gliner-multitask-large-v0.5", "threshold": 0.7, "use_mapping": False},

    {"model": "urchade/gliner_small-v2.1", "threshold": 0.3, "use_mapping": False},
    {"model": "urchade/gliner_small-v2.1", "threshold": 0.5, "use_mapping": False},
    {"model": "urchade/gliner_small-v2.1", "threshold": 0.7, "use_mapping": False},

    {"model": "nvidia/gliner-PII", "threshold": 0.3, "use_mapping": False},
    {"model": "nvidia/gliner-PII", "threshold": 0.4, "use_mapping": False},
    {"model": "nvidia/gliner-PII", "threshold": 0.7, "use_mapping": False},

    {"model": "knowledgator/modern-gliner-bi-large", "threshold": 0.3, "use_mapping": False},
    {"model": "knowledgator/modern-gliner-bi-large", "threshold": 0.5, "use_mapping": False},
    {"model": "knowledgator/modern-gliner-bi-large", "threshold": 0.7, "use_mapping": False},

    #com tags
    {"model": "knowledgator/gliner-multitask-large-v0.5", "threshold": 0.3, "use_mapping": True},
    {"model": "knowledgator/gliner-multitask-large-v0.5", "threshold": 0.5, "use_mapping": True},
    {"model": "knowledgator/gliner-multitask-large-v0.5", "threshold": 0.7, "use_mapping": True},

    {"model": "urchade/gliner_small-v2.1", "threshold": 0.3, "use_mapping": True},
    {"model": "urchade/gliner_small-v2.1", "threshold": 0.5, "use_mapping": True},
    {"model": "urchade/gliner_small-v2.1", "threshold": 0.7, "use_mapping": True},

    {"model": "nvidia/gliner-PII", "threshold": 0.3, "use_mapping": True},
    {"model": "nvidia/gliner-PII", "threshold": 0.4, "use_mapping": True},
    {"model": "nvidia/gliner-PII", "threshold": 0.7, "use_mapping": True},

    {"model": "knowledgator/modern-gliner-bi-large", "threshold": 0.3, "use_mapping": True},
    {"model": "knowledgator/modern-gliner-bi-large", "threshold": 0.5, "use_mapping": True},
    {"model": "knowledgator/modern-gliner-bi-large", "threshold": 0.7, "use_mapping": True},
]

def bio_to_entity_list(tokens, tags):
    entities = []
    chunk = []
    label = None
    for t, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if chunk: entities.append({"text": " ".join(chunk), "label": label})
            chunk = [t]
            label = tag[2:]
        elif tag.startswith('I-') and label == tag[2:]:
            chunk.append(t)
        else:
            if chunk: entities.append({"text": " ".join(chunk), "label": label})
            chunk = []
            label = None
    if chunk: entities.append({"text": " ".join(chunk), "label": label})
    return entities

def align_gliner_to_bio(tokens, entities, use_mapping):
    tags = ['O'] * len(tokens)
    text = ""
    token_spans = []

    for token in tokens:
        start = len(text)
        if start > 0:
            text += " "
            start += 1
        end = start + len(token)
        text += token
        token_spans.append((start, end))

    for idx, (tok_start, tok_end) in enumerate(token_spans):
        for ent in entities:
            if max(tok_start, ent['start']) < min(tok_end, ent['end']):
                label_found = ent['label']

                if use_mapping:
                    final_label = REVERSE_MAPPING.get(label_found, label_found)
                else:
                    final_label = label_found

                if abs(tok_start - ent['start']) <= 1:
                    prefix = "B-"
                else:
                    prefix = "I-"

                if tags[idx] == 'O':
                    tags[idx] = f"{prefix}{final_label}"
                break
    return tags, text

def get_labels_for_config(loader, use_mapping):
    all_labels = set()
    for f in range(1, TOTAL_FOLDS + 1):
        _, bio_tags = loader.load_fold(f)
        for seq in bio_tags:
            for t in seq:
                if t != 'O':
                    all_labels.add(t[2:])

    raw_list = sorted(list(all_labels))

    if use_mapping:
        return [TAG_MAPPING.get(tag, tag) for tag in raw_list]
    else:
        return raw_list

def run_single_config(loader, config):
    model_name = config['model']
    thresh = config['threshold']
    use_mapping = config['use_mapping']

    labels_list = get_labels_for_config(loader, use_mapping)

    map_suffix = "_MAPPED" if use_mapping else "_RAW"
    safe_name = model_name.split('/')[-1].replace('.', '')
    run_id = f"{safe_name}_T{str(thresh).replace('.', '')}{map_suffix}"

    print(f"\n>>> INICIANDO: {model_name} | Threshold: {thresh} | Mapping: {use_mapping}")

    try:
        model = GLiNER.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception as e:
        print(f"ERRO ao carregar {model_name}: {e}")
        return None

    base_dir = os.path.join("Resultados_GLiNER_Benchmark", run_id)
    os.makedirs(base_dir, exist_ok=True)

    fold_metrics_summary = []

    for fold in range(1, TOTAL_FOLDS + 1):
        test_sents, true_bio = loader.load_fold(fold)
        if not test_sents: continue
        if LIMIT_SAMPLES:
            test_sents = test_sents[:LIMIT_SAMPLES]
            true_bio = true_bio[:LIMIT_SAMPLES]

        fold_dir = os.path.join(base_dir, f"Fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        pred_bio_list = []
        feedback_rows = []
        desc_bar = f"Fold {fold}/{TOTAL_FOLDS}"

        for i, tokens in tqdm(enumerate(test_sents), total=len(test_sents), desc=desc_bar, leave=False):
            text_input = " ".join(tokens)
            try:
                ents = model.predict_entities(text_input, labels_list, threshold=thresh)
            except:
                ents = []

            tags, _ = align_gliner_to_bio(tokens, ents, use_mapping)
            pred_bio_list.append(tags)

            true_ents_obj = bio_to_entity_list(tokens, true_bio[i])
            pred_ents_obj = []
            for e in ents:
                lbl = e['label']
                if use_mapping:
                    lbl = REVERSE_MAPPING.get(lbl, lbl)
                pred_ents_obj.append({'text': e['text'], 'label': lbl})

            json_true = json.dumps(sorted(true_ents_obj, key=lambda x: x['text']), ensure_ascii=False)
            json_pred = json.dumps(sorted(pred_ents_obj, key=lambda x: x['text']), ensure_ascii=False)

            status = "PERFEITO" if json_true == json_pred else "DIVERGENTE"
            feedback_rows.append({
                "texto": text_input,
                "reais": json_true.replace('"', "'"),
                "previstos": json_pred.replace('"', "'"),
                "status": status
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

        fold_metrics_summary.append({
            "fold": fold,
            "precision": report.get('micro avg', {}).get('precision', 0),
            "recall": report.get('micro avg', {}).get('recall', 0),
            "f1": report.get('micro avg', {}).get('f1-score', 0)
        })

    df_folds = pd.DataFrame(fold_metrics_summary)
    if df_folds.empty: return None

    avg_results = {
        "Modelo": model_name,
        "Threshold": thresh,
        "Mapping": "Sim" if use_mapping else "Nao",
        "F1_Medio": df_folds['f1'].mean(),
        "Precision_Media": df_folds['precision'].mean(),
        "Recall_Medio": df_folds['recall'].mean(),
    }
    df_folds.to_csv(os.path.join(base_dir, "RELATORIO_FOLDS_RESUMO.csv"), index=False)
    return avg_results

def run_benchmark():
    loader = DataLoader(PASTA_DATASET)
    global_results = []
    output_folder = "Resultados_GLiNER_Benchmark"
    os.makedirs(output_folder, exist_ok=True)
    path_progresso = os.path.join(output_folder, "PROGRESSO_PARCIAL.csv")

    print("--- Iniciando Bateria de Testes Otimizada ---")

    for config in GLINER_CONFIGS:
        start_run = time.time()
        result = run_single_config(loader, config)
        duration = time.time() - start_run

        if result:
            result['Tempo_Segundos'] = round(duration, 2)
            global_results.append(result)
            pd.DataFrame(global_results).to_csv(path_progresso, index=False)
            print(f"   -> F1: {result['F1_Medio']:.4f} | Tempo: {duration:.1f}s")
        else:
            print("   -> Falha na execução deste modelo.")

    print("\n" + "="*60 + "\nRANKING FINAL\n" + "="*60)
    if global_results:
        df_final = pd.DataFrame(global_results).sort_values(by="F1_Medio", ascending=False)
        df_final.to_csv(os.path.join(output_folder, "RANKING_FINAL_GLINER.csv"), index=False)
        print(df_final.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()

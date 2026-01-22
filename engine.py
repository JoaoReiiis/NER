import os
import glob
import json
import ollama
import random
from config import get_system_prompt, get_user_prompt, OLLAMA_OPTS

class DataLoader:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.all_tags = set()

    def load_fold(self, fold_num):
        pattern = os.path.join(self.base_folder, f"*particao_{fold_num}.*")
        files = glob.glob(pattern)

        if not files: return [], []

        filepath = files[0]
        sentences = []
        tags_bio_list = []
        curr_sent, curr_tags = [], []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if curr_sent:
                        sentences.append(curr_sent)
                        tags_bio_list.append(curr_tags)
                        curr_sent, curr_tags = [], []
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    curr_sent.append(parts[0])
                    raw_tag = parts[-1]
                    if raw_tag == 'O':
                        curr_tags.append('O')
                    else:
                        prefix = raw_tag[0:2]
                        label = raw_tag[2:].upper()
                        curr_tags.append(f"{prefix}{label}")
                        self.all_tags.add(label)

        if curr_sent:
            sentences.append(curr_sent)
            tags_bio_list.append(curr_tags)
        return sentences, tags_bio_list

    def load_training_context(self, test_fold_idx, total_folds):
        all_sents = []
        all_bios = []

        for f in range(1, total_folds + 1):
            if f == test_fold_idx: continue

            s, b = self.load_fold(f)
            all_sents.extend(s)
            all_bios.extend(b)

        return all_sents, all_bios

class LLMEngine:
    def __init__(self, model_name):
        self.model = model_name

    def _format_few_shot(self, examples_list):
        if not examples_list: return ""

        block = ""
        for i, (tokens, bio_tags) in enumerate(examples_list):
            text = " ".join(tokens)
            entities_json = self.bio_to_json(tokens, bio_tags)
            json_str = json.dumps(entities_json, ensure_ascii=False)

            block += f"\nExample {i+1}:\nInput: \"{text}\"\nOutput: {json_str}\n"
        return block

    def predict(self, tokens, valid_tags_list, few_shot_examples=None):
        text = " ".join(tokens)

        examples_block = self._format_few_shot(few_shot_examples)

        sys_msg = get_system_prompt(valid_tags_list, examples_block)
        usr_msg = get_user_prompt(text)
        allowed_labels = set(t.upper() for t in valid_tags_list)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': usr_msg}],
                options=OLLAMA_OPTS,
                format='json',
                keep_alive='5m'
            )

            content = response['message']['content']
            try:
                raw_entities = json.loads(content)
            except:
                raw_entities = []

            if isinstance(raw_entities, dict): raw_entities = [raw_entities]
            if not isinstance(raw_entities, list): raw_entities = []

            valid_entities = []
            hallucination_detected = False

            for item in raw_entities:
                if isinstance(item, dict) and 'text' in item and 'label' in item:
                    clean_label = str(item['label']).upper()
                    if clean_label in allowed_labels:
                        valid_entities.append({"text": str(item['text']), "label": clean_label})
                    else:
                        hallucination_detected = True

            bio_tags = self._json_to_bio(tokens, valid_entities)
            return valid_entities, bio_tags, hallucination_detected

        except Exception as e:
            print(f"Erro: {e}")
            return [], ['O'] * len(tokens), False

    def _json_to_bio(self, tokens, entities):
        tags = ['O'] * len(tokens)
        if not entities: return tags
        for ent in entities:
            ent_parts = ent['text'].split()
            label = ent['label']
            size = len(ent_parts)
            if size == 0: continue
            for i in range(len(tokens) - size + 1):
                sub = tokens[i : i + size]
                if [t.lower() for t in sub] == [e.lower() for e in ent_parts]:
                    tags[i] = f"B-{label}"
                    for j in range(1, size): tags[i+j] = f"I-{label}"
                    break
        return tags

    def bio_to_json(self, tokens, tags):
        entities = []
        chunk, label = [], None
        for t, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                if chunk: entities.append({"text": " ".join(chunk), "label": label})
                chunk = [t]
                label = tag[2:]
            elif tag.startswith('I-') and label == tag[2:]:
                chunk.append(t)
            else:
                if chunk: entities.append({"text": " ".join(chunk), "label": label})
                chunk, label = [], None
        if chunk: entities.append({"text": " ".join(chunk), "label": label})
        return entities

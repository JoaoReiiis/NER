
PASTA_DATASET = "cachacaNER"
TOTAL_FOLDS = 10
LIMIT_SAMPLES = None
BENCHMARK_MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "llama3.2:1b",
    "llama3.2:3b"
]

BENCHMARK_SHOTS = [0,3,5]
OLLAMA_OPTS = {
    'temperature': 0.0,
    'num_ctx': 2048,
    'seed': 42,
    'num_predict': 512,
}

TAG_DEFINITIONS = """
1. **NOME_BEBIDA:** O nome próprio da marca da cachaça. Extraia apenas a marca (ex: "51", "Velho Barreiro"), evite termos genéricos.
2. **NOME_PESSOA:** Nomes de indivíduos específicos citados (produtores, mestres alambiqueiros, fundadores).
3. **NOME_LOCAL:** Entidades geográficas específicas de origem ou produção (Fazendas, Sítios, Cidades, Estados).
4. **NOME_ORGANIZACAO:** Pessoas jurídicas, empresas fabricantes, cooperativas ou associações.
5. **GRADUACAO_ALCOOLICA:** O valor numérico do teor alcoólico seguido da unidade (ex: "40%", "42% vol").
6. **TEMPO_ARMAZENAMENTO:** A duração explícita de envelhecimento ou descanso (ex: "2 anos", "6 meses").
7. **TIPO_MADEIRA:** O nome da madeira específica utilizada no envelhecimento (ex: "Carvalho", "Amburana", "Bálsamo").
8. **VOLUME:** A capacidade volumétrica da garrafa ou recipiente (ex: "700ml", "1 Litro", "600ml").
9. **PRECO:** Valores monetários explícitos associados ao produto (ex: "R$ 50,00", "120 reais").
10. **CLASSIFICACAO_BEBIDA:** A designação legal ou comercial de qualidade (ex: "Ouro", "Prata", "Premium", "Extra Premium").
11. **CARACTERISTICA_SENSORIAL_*:** Adjetivos descritivos específicos de Sabor, Aroma, Cor, Textura ou Retrogosto.
"""

JSON_SCHEMA = """[
  {"texto": "trecho exato do texto original", "rotulo": "CATEGORIA_DA_TAG"}
]"""

def get_system_prompt(valid_tags, examples_block=""):
    tags_list = ", ".join(sorted(valid_tags))

    examples_section = ""
    if examples_block:
        examples_section = f"\n### EXEMPLOS DE REFERÊNCIA (FEW-SHOT)\nAnalise como as entidades foram extraídas nestes exemplos e siga o padrão:\n{examples_block}\n"

    return f"""
Você é um Especialista em Processamento de Linguagem Natural focado em Extração de Entidades (NER) para o domínio de Cachaças.
Sua tarefa é analisar o texto de entrada e extrair entidades que correspondam estritamente às definições fornecidas.

### FORMATO DE SAÍDA OBRIGATÓRIO
Retorne APENAS um JSON válido. Não inclua markdown, explicações ou texto adicional.
Schema: {JSON_SCHEMA}

### DEFINIÇÕES E REGRAS
{TAG_DEFINITIONS}
{examples_section}

### RESTRIÇÕES E DIRETRIZES
1. **Exatidão:** O campo "text" deve ser uma cópia exata do trecho original. Não corrija ortografia ou capitalização.
2. **Contexto:** Não extraia palavras comuns/genéricas se não forem entidades nomeadas (ex: não extraia "madeira" se não especificar qual madeira).
3. **Limpeza:** Não inclua pontuação (vírgulas, pontos) que não pertença à entidade.
4. **Vazio:** Se nenhuma entidade correspondente for encontrada, retorne estritamente: []

### RÓTULOS (LABELS) PERMITIDOS
Use apenas estes rótulos:
{tags_list}
"""

def get_user_prompt(text):
    return f'Input: "{text}"\nOutput JSON:'

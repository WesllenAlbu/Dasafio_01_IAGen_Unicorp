# 🎯 GLPI Triagem Inteligente

Sistema de análise e triagem de chamados GLPI do ERP Protheus com Inteligência Artificial.


## 📋 Visão Geral

Esta aplicação lê um arquivo CSV exportado do GLPI e oferece quatro funcionalidades principais: triagem de chamados com análise por IA, identificação de chamados repetidos, análise de setores com mais aberturas e um sistema de consulta em linguagem natural sobre os dados.


## 🤖 Métodos de IA Utilizados

### 1. Geração de Texto com LLM (Large Language Model)

**Onde é usado:** Abas "Triagem por Chamado", "Chamados Repetidos" e "Setor com Mais Aberturas"

**Como funciona:**

O sistema monta um prompt estruturado com os dados do chamado ou grupo selecionado e envia para o modelo **GPT-4o-mini** da OpenAI via API. O modelo recebe a instrução de retornar exclusivamente um JSON válido com campos como resumo, sentimento, complexidade, impacto, ações sugeridas e rascunho de resposta.

**Técnica:** Prompt Engineering com saída estruturada em JSON. O modelo é instruído a atuar como especialista em suporte ERP Protheus e a responder apenas com base nas informações fornecidas no prompt.


### 2. RAG (Retrieval Augmented Generation)

**Onde é usado:** Aba "💬 Consulta aos Dados"

**Como funciona em três etapas:**

**Etapa 1: Indexação**

Cada chamado do CSV é convertido em um texto estruturado contendo ID, título, prioridade, módulo, tipo, técnico, setor e descrição. Esses textos são chamados de *chunks*.

**Etapa 2: Busca Semântica**

Quando o usuário faz uma pergunta, o modelo **all-MiniLM-L6-v2** (SentenceTransformer) transforma tanto a pergunta quanto todos os chunks em vetores numéricos chamados *embeddings*. Em seguida, o sistema calcula a **similaridade do cosseno** entre o vetor da pergunta e os vetores de todos os chamados, selecionando os 10 chunks com maior similaridade semântica.

**Etapa 3: Geração da Resposta**

Os 10 chunks mais relevantes são enviados como contexto para o **GPT-4o-mini**, que gera uma resposta baseada exclusivamente nesses trechos selecionados.

**Por que RAG?** Em vez de enviar todos os 633 chamados para o modelo (o que seria caro e lento), o sistema seleciona apenas os trechos mais relevantes para a pergunta. Isso torna as respostas mais precisas e o processamento mais eficiente.


## 🔬 Modelos e Bibliotecas

| Função | Modelo / Biblioteca |
|---|---|
| Geração de análise e respostas | GPT-4o-mini (OpenAI) |
| Embeddings para busca semântica | all-MiniLM-L6-v2 (SentenceTransformer) |
| Cálculo de similaridade | Cosine Similarity (scikit-learn) |
| Interface web | Streamlit |
| Manipulação de dados | Pandas / NumPy |


## ⚙️ Como Rodar

**1. Instale as dependências:**

```bash
pip install streamlit pandas openai python-dotenv sentence-transformers scikit-learn
```

**2. Configure o arquivo `.env` na mesma pasta:**

```
OPENAI_API_KEY=sua_chave_aqui
```

**3. Execute a aplicação:**

```bash
python -m streamlit run app.py
```

**4. Faça upload do CSV exportado do GLPI** pelo menu lateral e comece a explorar.


## 📁 Estrutura Esperada do CSV

O arquivo CSV deve ser exportado diretamente do GLPI com separador ponto e vírgula (`;`) e conter colunas como ID, Título, Status, Prioridade, Categoria, Descrição, Localização, Atribuído (Técnico), Requerente e Estatísticas de fechamento.


## 🗂️ Abas da Aplicação

**Triagem por Chamado**
Visualize gráficos gerais, filtre chamados por prioridade e módulo, selecione um chamado na tabela e processe a análise com IA para obter resumo, sentimento, complexidade, impacto, ações sugeridas e rascunho de resposta.

**Chamados Repetidos**
Identifica automaticamente combinações de módulo e tipo de solicitação com mais de uma ocorrência. Selecione um grupo e a IA analisa a causa raiz, o padrão identificado e sugere ações preventivas.

**Setor com Mais Aberturas**
Ranking de setores por volume de chamados com gráficos de detalhe por módulo, prioridade e tipo. A IA identifica o motivo principal das aberturas e o perfil do setor.

**Consulta aos Dados**
Chat em linguagem natural com busca semântica sobre todos os chamados do CSV. Faça perguntas livres como "Qual técnico tem mais chamados de alta prioridade?" e receba respostas baseadas nos dados reais.
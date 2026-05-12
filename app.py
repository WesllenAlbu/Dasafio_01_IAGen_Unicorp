import os
import json
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "").strip().strip("'").strip('"')
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="GLPI Triagem Inteligente", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: Arial, sans-serif; background-color: #ffffff; color: #222222; }
    [data-testid="stSidebar"] { background-color: #f5f5f5; }
    .stButton > button { background-color: #0066cc; color: white; border-radius: 4px; border: none; padding: 8px 20px; }
    .stButton > button:hover { background-color: #0052a3; }
    .ticket-selecionado {
        background-color: #f0f4ff; border: 1px solid #0066cc44;
        border-radius: 6px; padding: 10px 14px;
        font-size: 15px; font-weight: 600; color: #0066cc; margin-bottom: 12px;
    }
    .ticket-vazio {
        background-color: #f9f9f9; border: 1px solid #cccccc;
        border-radius: 6px; padding: 10px 14px;
        font-size: 14px; color: #999999; margin-bottom: 12px; font-style: italic;
    }
    /* Botão Enviar da aba4 vermelho */
    div[data-testid="column"] .stButton > button[kind="primary"],
    #btn_enviar_aba4 { background-color: #cc0000 !important; }
</style>
""", unsafe_allow_html=True)

SCROLL_JS = """
<script>
    setTimeout(function() {
        const els = window.parent.document.querySelectorAll('h3');
        for (const el of els) {
            if (el.innerText.includes('Análise com IA')) {
                el.scrollIntoView({behavior: 'smooth', block: 'start'});
                break;
            }
        }
    }, 300);
</script>
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_modulo(cat):
    if not isinstance(cat, str): return "Outros"
    m = re.search(r"> (\d+-[^>]+)$", cat)
    return m.group(1).strip() if m else cat.split(">")[-1].strip()

def extract_desc(desc):
    if not isinstance(desc, str): return ""
    m = re.search(r"(?:6\)|8\))\s*DESCREVA.?:\s\n(.*?)(?:\n\d+\)|\Z)", desc, re.IGNORECASE | re.DOTALL)
    if m:
        text = re.sub(r"#[a-f0-9\-]+#[\d.]+", "", m.group(1)).strip()
        text = re.sub(r"\[image\]", "", text).strip()
        return text[:400]
    clean = re.sub(r"#[a-f0-9\-]+#[\d.]+", "", desc)
    return clean[:300].strip()

def extract_tipo(desc):
    if not isinstance(desc, str): return ""
    m = re.search(r"3\).?OPÇÃO.?:\s*-\.(.*?)\.", desc, re.IGNORECASE)
    return m.group(1).strip() if m else ""

@st.cache_data
def carregar_csv(arquivo):
    df = pd.read_csv(arquivo, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
    df["modulo"]           = df["Categoria"].apply(extract_modulo)
    df["desc_curta"]       = df["Descrição"].apply(extract_desc)
    df["tipo_solicitacao"] = df["Descrição"].apply(extract_tipo)
    df["setor"]            = df["Localização"].fillna("Não informado")
    return df

# ── RAG ────────────────────────────────────────────────────────────────────────

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def carregar_modelo():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def montar_chunks(shape):
    chunks = []
    for _, row in df.iterrows():
        chunks.append(
            f"ID:{row.get('ID','')} | {str(row.get('Título',''))[:60]} | "
            f"Prioridade:{row.get('Prioridade','')} | Módulo:{row.get('modulo','')} | "
            f"Tipo:{row.get('tipo_solicitacao','')} | Técnico:{row.get('Atribuído - Técnico','')} | "
            f"Setor:{row.get('setor','')} | Tempo:{row.get('Estatísticas - Hora de fechamento','')} | "
            f"Desc:{str(row.get('desc_curta',''))[:120]}"
        )
    return chunks

@st.cache_data
def codificar_chunks(shape):
    return carregar_modelo().encode(montar_chunks(shape))

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuração")
    arquivo = st.file_uploader("Carregar CSV do GLPI", type=["csv"])

if not arquivo:
    st.title("🎯 GLPI Triagem Inteligente")
    st.info("Faça upload do CSV do GLPI no menu lateral para começar.")
    st.stop()

df = carregar_csv(arquivo)

for k in ["id_clicado", "grupo_clicado", "setor_clicado", "historico", "ultima_resposta"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "historico" else []

st.title("🎯 GLPI Triagem Inteligente")
st.caption(f"{len(df)} chamados carregados")

aba1, aba2, aba3, aba4 = st.tabs(["Triagem por Chamado", "Chamados Repetidos", "Setor com Mais Aberturas", "💬 Consulta aos Dados"])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — TRIAGEM POR CHAMADO
# ══════════════════════════════════════════════════════════════════════════════

with aba1:
    st.subheader("Triagem Inteligente de Chamados")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("*Por Prioridade*")
        pc = df["Prioridade"].value_counts().reset_index()
        pc.columns = ["Prioridade","Qtd"]
        st.bar_chart(pc.set_index("Prioridade"), height=200)
    with g2:
        st.markdown("*Por Tipo de Solicitação*")
        tc = df[df["tipo_solicitacao"] != ""]["tipo_solicitacao"].value_counts().reset_index()
        tc.columns = ["Tipo","Qtd"]
        if not tc.empty:
            st.bar_chart(tc.set_index("Tipo"), height=200)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("*Top 8 Módulos*")
        mc = df["modulo"].value_counts().head(8).reset_index()
        mc.columns = ["Módulo","Qtd"]
        st.bar_chart(mc.set_index("Módulo"), height=200)
    with g4:
        st.markdown("*Top 8 Técnicos*")
        tec = df["Atribuído - Técnico"].value_counts().head(8).reset_index()
        tec.columns = ["Técnico","Qtd"]
        st.bar_chart(tec.set_index("Técnico"), height=200)

    st.markdown("---")

    busca = st.text_input("Buscar por ID, título ou requerente:")
    col1, col2 = st.columns(2)
    with col1:
        filtro_prio = st.selectbox("Prioridade", ["Todas"] + sorted(df["Prioridade"].dropna().unique().tolist()))
    with col2:
        filtro_mod = st.selectbox("Módulo", ["Todos"] + sorted(df["modulo"].dropna().unique().tolist()))

    mask = pd.Series([True] * len(df))
    if busca.strip():
        q = busca.strip().lower()
        mask &= (
            df["Título"].str.lower().str.contains(q, na=False) |
            df["ID"].astype(str).str.contains(q, na=False) |
            df["Requerente - Requerente"].str.lower().str.contains(q, na=False)
        )
    if filtro_prio != "Todas":
        mask &= df["Prioridade"] == filtro_prio
    if filtro_mod != "Todos":
        mask &= df["modulo"] == filtro_mod

    filtrado = df[mask].reset_index(drop=True)
    st.caption(f"{len(filtrado)} chamado(s) encontrado(s)")

    if not filtrado.empty:
        tabela = filtrado[["ID","Título","Prioridade","modulo","tipo_solicitacao",
                            "Atribuído - Técnico","Requerente - Requerente",
                            "Estatísticas - Hora de fechamento"]].copy()
        tabela = tabela.rename(columns={
            "modulo":"Módulo","tipo_solicitacao":"Tipo",
            "Atribuído - Técnico":"Técnico",
            "Requerente - Requerente":"Requerente",
            "Estatísticas - Hora de fechamento":"Tempo Fechamento"
        })
        ev1 = st.dataframe(
            tabela, use_container_width=True, hide_index=True, height=400,
            on_select="rerun", selection_mode="single-row",
            column_config={"ID": st.column_config.TextColumn("ID", width="small")}
        )
        linhas1 = ev1.selection.get("rows", [])
        if linhas1:
            id_novo = str(filtrado.iloc[linhas1[0]]["ID"])
            if id_novo != st.session_state.id_clicado:
                st.session_state.id_clicado = id_novo

    if st.session_state.id_clicado:
        st.markdown(SCROLL_JS, unsafe_allow_html=True)

    st.divider()
    st.subheader("Análise com IA")

    if filtrado.empty:
        st.warning("Nenhum chamado para analisar.")
    else:
        st.markdown("*Chamado selecionado:*")
        ticket = None
        if st.session_state.id_clicado:
            match = filtrado[filtrado["ID"].astype(str) == st.session_state.id_clicado]
            if not match.empty:
                ticket = match.iloc[0]
                st.markdown(f'<div class="ticket-selecionado">🎯 #{ticket["ID"]} — {ticket["Título"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ticket-vazio">Nenhum chamado selecionado. Selecione em uma linha da tabela acima.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ticket-vazio">Nenhum chamado selecionado. Selecione em uma linha da tabela acima.</div>', unsafe_allow_html=True)

        if ticket is not None:
            st.markdown(f"*Descrição:* {ticket.get('desc_curta','') or 'Sem descrição.'}")

            if st.button("Processar Ticket com IA", key="btn_aba1"):
                prompt = f"""Analise o chamado GLPI abaixo e retorne APENAS um JSON válido com as chaves:
- "resumo": resumo do problema em até 2 frases
- "sentimento": (neutro, irritado, urgente, tranquilo)
- "complexidade": (baixa, média, alta)
- "impacto": (operacional, financeiro, fiscal, infraestrutura, acesso)
- "acoes_sugeridas": lista de strings com próximos passos
- "resposta_rascunho": mensagem empática e profissional para o requerente (máx 3 frases)
- "palavras_chave": lista de até 4 palavras-chave

Chamado:
- ID: {ticket.get('ID','')}
- Título: {ticket.get('Título','')}
- Módulo: {ticket.get('modulo','')}
- Tipo: {ticket.get('tipo_solicitacao','Não informado')}
- Prioridade: {ticket.get('Prioridade','')}
- Requerente: {ticket.get('Requerente - Requerente','')}
- Técnico: {ticket.get('Atribuído - Técnico','')}
- Localização: {ticket.get('Localização','')}
- Tempo de fechamento: {ticket.get('Estatísticas - Hora de fechamento','')}
- Descrição: {ticket.get('desc_curta','')}"""

                with st.spinner("Processando com IA..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Você é um especialista em suporte ERP Protheus. Retorne apenas JSON válido, sem markdown."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2
                    )
                    try:
                        result = json.loads(resp.choices[0].message.content)

                        st.subheader("📊 Dashboard Rápido")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Sentimento",   result.get("sentimento","").capitalize())
                        col_b.metric("Complexidade", result.get("complexidade","").capitalize())
                        col_c.metric("Impacto",      result.get("impacto","").capitalize())

                        st.info("📝 *Resumo:* " + result.get("resumo",""))

                        st.markdown("*Ações Sugeridas:*")
                        for i, a in enumerate(result.get("acoes_sugeridas",[]), 1):
                            st.write(f"{i}. {a}")

                        st.info("✉️ *Rascunho de Resposta:*\n" + result.get("resposta_rascunho",""))

                        kws = result.get("palavras_chave",[])
                        if kws:
                            st.markdown("*Palavras-chave:* " + " · ".join([f"{k}" for k in kws]))

                        with st.expander("JSON completo"):
                            st.json(result)

                    except Exception as e:
                        st.error(f"Falha ao parsear JSON: {e}\nRaw: {resp.choices[0].message.content}")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — CHAMADOS REPETIDOS
# ══════════════════════════════════════════════════════════════════════════════

with aba2:
    st.subheader("Incidência de Chamados Repetidos pelo Mesmo Problema")

    grupo = (
        df.groupby(["modulo","tipo_solicitacao"])
        .size()
        .reset_index(name="total")
        .sort_values("total", ascending=False)
    )
    grupo = grupo[grupo["total"] > 1].reset_index(drop=True)

    if not grupo.empty:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("*Top 10 Módulos com mais repetições*")
            mod_rep = grupo.groupby("modulo")["total"].sum().reset_index().sort_values("total", ascending=False).head(10)
            mod_rep.columns = ["Módulo","Total"]
            st.bar_chart(mod_rep.set_index("Módulo"), height=200)
        with g2:
            st.markdown("*Top 8 Grupos mais recorrentes*")
            top_g = grupo.head(8).copy()
            top_g["label"] = top_g["modulo"].str[:15] + " | " + top_g["tipo_solicitacao"].str[:12]
            st.bar_chart(top_g.set_index("label")["total"].rename("Ocorrências"), height=200)

        g3, g4 = st.columns(2)
        with g3:
            st.markdown("*Prioridade nos chamados repetidos*")
            ids_rep = []
            for _, rg in grupo.iterrows():
                ids_rep += df[(df["modulo"] == rg["modulo"]) & (df["tipo_solicitacao"] == rg["tipo_solicitacao"])]["ID"].tolist()
            df_rep = df[df["ID"].isin(ids_rep)]
            if not df_rep.empty:
                pr = df_rep["Prioridade"].value_counts().reset_index()
                pr.columns = ["Prioridade","Qtd"]
                st.bar_chart(pr.set_index("Prioridade"), height=200)
        with g4:
            st.markdown("*Tipo de Solicitação recorrente*")
            tipo_rep = grupo[grupo["tipo_solicitacao"] != ""].groupby("tipo_solicitacao")["total"].sum().reset_index().sort_values("total", ascending=False)
            tipo_rep.columns = ["Tipo","Total"]
            if not tipo_rep.empty:
                st.bar_chart(tipo_rep.set_index("Tipo"), height=200)

        st.markdown("---")

    st.caption(f"{len(grupo)} combinações de módulo + tipo com mais de 1 chamado")
    if not grupo.empty:
        ev2 = st.dataframe(
            grupo.rename(columns={"modulo":"Módulo","tipo_solicitacao":"Tipo","total":"Qtd Chamados"}),
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row"
        )
        linhas2 = ev2.selection.get("rows", [])
        if linhas2:
            idx_novo = int(linhas2[0])
            if idx_novo != st.session_state.grupo_clicado:
                st.session_state.grupo_clicado = idx_novo

    if st.session_state.grupo_clicado is not None:
        st.markdown(SCROLL_JS, unsafe_allow_html=True)

    st.divider()
    st.subheader("Análise com IA — Causa Raiz da Recorrência")

    if grupo.empty:
        st.info("Nenhum grupo recorrente encontrado.")
    else:
        st.markdown("*Grupo selecionado:*")
        row_g = None
        if st.session_state.grupo_clicado is not None and st.session_state.grupo_clicado < len(grupo):
            row_g = grupo.iloc[st.session_state.grupo_clicado]
            label_g = f"{row_g['modulo']} | {row_g['tipo_solicitacao']} ({row_g['total']} chamados)"
            st.markdown(f'<div class="ticket-selecionado">🎯 {label_g}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ticket-vazio">Nenhum grupo selecionado — clique em uma linha da tabela acima.</div>', unsafe_allow_html=True)

        if row_g is not None:
            chamados_grupo = df[
                (df["modulo"] == row_g["modulo"]) &
                (df["tipo_solicitacao"] == row_g["tipo_solicitacao"])
            ][["ID","Título","desc_curta","Prioridade","Requerente - Requerente"]].head(10)

            st.caption(f"{row_g['total']} chamados neste grupo — exibindo até 10")
            st.dataframe(chamados_grupo.rename(columns={
                "desc_curta":"Descrição","Requerente - Requerente":"Requerente"
            }), use_container_width=True, hide_index=True)

            if st.button("Analisar Recorrência com IA", key="btn_aba2"):
                amostras = "\n".join([
                    f"- #{r['ID']}: {r['Título']} | {r['desc_curta'][:120]}"
                    for _, r in chamados_grupo.iterrows()
                ])
                prompt = f"""Analise os chamados abaixo que se repetem pelo mesmo problema no sistema Protheus ERP.
Retorne APENAS um JSON válido com as chaves:
- "causa_raiz_provavel": string com a causa raiz mais provável
- "padrao_identificado": string descrevendo o padrão comum entre os chamados
- "acoes_preventivas": lista de strings com ações para evitar recorrência
- "recomendacao_tecnica": string com recomendação para o time de TI
- "nivel_criticidade": (baixo, médio, alto, crítico)

Módulo: {row_g['modulo']}
Tipo: {row_g['tipo_solicitacao']}
Total de ocorrências: {row_g['total']}

Chamados:
{amostras}"""

                with st.spinner("Processando com IA..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Você é um especialista em suporte ERP Protheus. Retorne apenas JSON válido, sem markdown."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2
                    )
                    try:
                        result = json.loads(resp.choices[0].message.content)

                        st.subheader("📊 Dashboard Rápido")
                        st.metric("Nível de Criticidade", result.get("nivel_criticidade","").capitalize())
                        st.info("🔍 *Causa Raiz Provável:*\n" + result.get("causa_raiz_provavel",""))
                        st.info("📌 *Padrão Identificado:*\n" + result.get("padrao_identificado",""))

                        st.markdown("*Ações Preventivas:*")
                        for i, a in enumerate(result.get("acoes_preventivas",[]), 1):
                            st.write(f"{i}. {a}")

                        st.info("🔧 *Recomendação Técnica:*\n" + result.get("recomendacao_tecnica",""))

                        with st.expander("JSON completo"):
                            st.json(result)

                    except Exception as e:
                        st.error(f"Falha ao parsear JSON: {e}\nRaw: {resp.choices[0].message.content}")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — SETOR COM MAIS ABERTURAS
# ══════════════════════════════════════════════════════════════════════════════

with aba3:
    st.subheader("Setor com Mais Aberturas de Chamado e Motivo")

    por_setor = (
        df.groupby("setor")
        .size()
        .reset_index(name="total")
        .sort_values("total", ascending=False)
    ).reset_index(drop=True)

    st.markdown("*Top 10 Setores por volume*")
    st.bar_chart(por_setor.head(10).set_index("setor")["total"].rename("Chamados"), height=220)

    st.markdown("---")

    # Tabela PRIMEIRO — captura a seleção antes de renderizar os gráficos de detalhe
    ev3 = st.dataframe(
        por_setor.rename(columns={"setor":"Setor","total":"Qtd Chamados"}),
        use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key="tabela_setor"
    )
    linhas3 = ev3.selection.get("rows", [])
    if linhas3:
        setor_novo = str(por_setor.iloc[linhas3[0]]["setor"])
        if setor_novo != st.session_state.setor_clicado:
            st.session_state.setor_clicado = setor_novo

    # Gráficos de detalhe DEPOIS — agora session_state já tem o valor correto
    setor_atual = st.session_state.setor_clicado
    if setor_atual:
        st.markdown(f"*Detalhes — {setor_atual}*")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("*Módulos*")
            ms = df[df["setor"] == setor_atual]["modulo"].value_counts().head(8).reset_index()
            ms.columns = ["Módulo","Qtd"]
            if not ms.empty:
                st.bar_chart(ms.set_index("Módulo"), height=200)
        with g2:
            st.markdown("*Prioridade*")
            ps = df[df["setor"] == setor_atual]["Prioridade"].value_counts().reset_index()
            ps.columns = ["Prioridade","Qtd"]
            st.bar_chart(ps.set_index("Prioridade"), height=200)
        with g3:
            st.markdown("*Tipo de Solicitação*")
            ts = df[(df["setor"] == setor_atual) & (df["tipo_solicitacao"] != "")]["tipo_solicitacao"].value_counts().reset_index()
            ts.columns = ["Tipo","Qtd"]
            if not ts.empty:
                st.bar_chart(ts.set_index("Tipo"), height=200)

    if st.session_state.setor_clicado:
        st.markdown(SCROLL_JS, unsafe_allow_html=True)

    st.divider()
    st.subheader("Análise com IA — Por Que Este Setor Abre Tanto Chamado?")

    st.markdown("*Setor selecionado:*")
    if st.session_state.setor_clicado:
        st.markdown(f'<div class="ticket-selecionado">🎯 {st.session_state.setor_clicado}</div>', unsafe_allow_html=True)
        setor_escolhido = st.session_state.setor_clicado
    else:
        st.markdown('<div class="ticket-vazio">Nenhum setor selecionado — clique em uma linha da tabela acima.</div>', unsafe_allow_html=True)
        setor_escolhido = None

    if setor_escolhido:
        chamados_setor = df[df["setor"] == setor_escolhido][
            ["ID","Título","modulo","tipo_solicitacao","Prioridade","desc_curta","Requerente - Requerente"]
        ].head(15)

        total_setor = len(df[df["setor"] == setor_escolhido])
        st.caption(f"{total_setor} chamados neste setor — exibindo até 15")
        st.dataframe(chamados_setor.rename(columns={
            "modulo":"Módulo","tipo_solicitacao":"Tipo",
            "desc_curta":"Descrição","Requerente - Requerente":"Requerente"
        }), use_container_width=True, hide_index=True)

        if st.button("Analisar Setor com IA", key="btn_aba3"):
            amostras = "\n".join([
                f"- #{r['ID']}: [{r['modulo']}] {r['Título']} | {r['desc_curta'][:100]}"
                for _, r in chamados_setor.iterrows()
            ])
            dist_mod  = df[df["setor"] == setor_escolhido]["modulo"].value_counts().head(5).to_dict()
            dist_tipo = df[df["setor"] == setor_escolhido]["tipo_solicitacao"].value_counts().head(5).to_dict()

            prompt = f"""Analise os chamados abertos pelo setor abaixo no sistema Protheus ERP.
Retorne APENAS um JSON válido com as chaves:
- "motivo_principal": string com o principal motivo de tantos chamados neste setor
- "modulos_mais_problematicos": lista de strings com os módulos mais problemáticos
- "perfil_do_setor": string descrevendo o perfil de uso do sistema por este setor
- "acoes_sugeridas": lista de strings com ações para reduzir os chamados
- "necessidade_treinamento": (sim, não, parcial)
- "resumo_executivo": string com resumo para apresentar à gestão (máx 3 frases)

Setor: {setor_escolhido}
Total de chamados: {total_setor}
Módulos mais acionados: {json.dumps(dist_mod, ensure_ascii=False)}
Tipos de solicitação: {json.dumps(dist_tipo, ensure_ascii=False)}

Amostra de chamados:
{amostras}"""

            with st.spinner("Processando com IA..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Você é um especialista em suporte ERP Protheus. Retorne apenas JSON válido, sem markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                try:
                    result = json.loads(resp.choices[0].message.content)

                    st.subheader("📊 Dashboard Rápido")
                    st.metric("Necessidade de Treinamento", result.get("necessidade_treinamento","").capitalize())
                    st.info("🏢 *Motivo Principal:*\n" + result.get("motivo_principal",""))
                    st.info("👤 *Perfil do Setor:*\n" + result.get("perfil_do_setor",""))

                    mods = result.get("modulos_mais_problematicos",[])
                    if mods:
                        st.markdown("*Módulos Mais Problemáticos:*")
                        for m in mods:
                            st.write(f"- {m}")

                    st.markdown("*Ações Sugeridas:*")
                    for i, a in enumerate(result.get("acoes_sugeridas",[]), 1):
                        st.write(f"{i}. {a}")

                    st.info("📋 *Resumo Executivo:*\n" + result.get("resumo_executivo",""))

                    with st.expander("JSON completo"):
                        st.json(result)

                except Exception as e:
                    st.error(f"Falha ao parsear JSON: {e}\nRaw: {resp.choices[0].message.content}")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 4 — CONSULTA AOS DADOS
# Mudanças: input no topo, nova pergunta substitui anterior, sem container
# ══════════════════════════════════════════════════════════════════════════════

with aba4:
    st.subheader("💬 Consulta aos Dados")
    st.caption("Faça uma pergunta sobre os chamados. A nova pergunta substitui a anterior.")

    with st.spinner("⏳ Indexando chamados... (apenas na primeira vez)"):
        chunks     = montar_chunks(df.shape)
        chunk_embs = codificar_chunks(df.shape)
        modelo     = carregar_modelo()

    # Input no topo
    pergunta = st.text_input("Faça sua pergunta:", placeholder="Ex: Qual técnico tem mais chamados de alta prioridade?", key="pergunta_aba4")

    if st.button("Enviar", key="btn_enviar_aba4"):
        if not pergunta.strip():
            st.warning("Digite uma pergunta antes de enviar.")
        else:
            query_emb = modelo.encode([pergunta])
            sims      = cosine_similarity(query_emb, chunk_embs)[0]
            top_idx   = np.argsort(sims)[-10:][::-1]
            trechos   = [f"Similaridade {sims[i]:.2f}: {chunks[i]}" for i in top_idx]
            context   = "\n".join([chunks[i] for i in top_idx])

            with st.spinner("Gerando resposta com IA..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um analista de suporte ERP Protheus. Responda apenas com base nos trechos fornecidos."
                        },
                        {
                            "role": "user",
                            "content": f"Contexto:\n{context}\nPergunta: {pergunta}"
                        }
                    ],
                    temperature=0.2
                )
                resposta = resp.choices[0].message.content

            # Substitui — guarda apenas a última pergunta e resposta
            st.session_state.ultima_resposta = {
                "pergunta": pergunta,
                "resposta": resposta,
                "trechos":  trechos
            }

    # Exibe a última pergunta e resposta no estilo chat
    if st.session_state.ultima_resposta:
        st.markdown("---")
        with st.chat_message("user"):
            st.write(st.session_state.ultima_resposta["pergunta"])
        with st.chat_message("assistant"):
            st.write(st.session_state.ultima_resposta["resposta"])
            with st.expander("📖 Trechos relevantes utilizados"):
                for t in st.session_state.ultima_resposta["trechos"]:
                    st.success(t)
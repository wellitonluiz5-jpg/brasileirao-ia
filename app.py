import datetime
import numpy as np
import pandas as pd
from scipy.stats import poisson
import streamlit as st

# Configuração da página em modo wide
st.set_page_config(
    page_title="Painel Pro - Brasileirão IA", page_icon="⚽", layout="wide"
)

# Estilização Ousada e Moderna com Cores Vivas (Tema Dark Profissional Neon)
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    h1, h2, h3 { color: #00ffcc !important; font-family: 'Inter', sans-serif; }
    .stMetric { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 15px; border-radius: 12px; border: 1px solid #374151; box-shadow: 0 4px 6px -1px rgba(0, 255, 204, 0.1); }
    .card-ia { background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 20px; border-radius: 12px; border: 2px solid #34d399; color: white; margin-bottom: 20px; box-shadow: 0 0 15px rgba(52, 211, 153, 0.4); }
    .card-palpite { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 6px solid #00ffcc; margin-bottom: 10px; color: #f3f4f6; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Painel Pro - Brasileirão IA")
st.markdown(
    "<p style='color: #9ca3af;'>Inteligência estatística aplicada a apostas"
    " esportivas (Série A & B)</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


@st.cache_data
def carregar_dados():
  try:
    return pd.read_csv("dados_brasileirao.csv")
  except FileNotFoundError:
    return None


df = carregar_dados()

if df is None:
  st.error(
      "⚠️ O arquivo 'dados_brasileirao.csv' não foi encontrado na pasta do"
      " projeto."
  )
else:
  lista_times = sorted(df["time"].unique())

  if "historico_palpites" not in st.session_state:
    st.session_state["historico_palpites"] = []

  st.sidebar.header("⚙️ Seleção de Equipes")
  time_casa = st.sidebar.selectbox("🏟️ Mandante", lista_times, index=0)
  time_visitante = st.sidebar.selectbox(
      "✈️ Visitante", lista_times, index=1 if len(lista_times) > 1 else 0
  )

  if time_casa == time_visitante:
    st.sidebar.warning("⚠️ Escolha times diferentes para o confronto!")
  else:
    dados_c = df[df["time"] == time_casa].iloc[0]
    dados_v = df[df["time"] == time_visitante].iloc[0]

    g_pro_c, g_contra_c = dados_c["gols_pro_casa"], dados_c["gols_contra_casa"]
    g_pro_f, g_contra_f = dados_v["gols_pro_fora"], dados_v["gols_contra_fora"]
    cantos_c, cartoes_c = dados_c["escanteios_pro"], dados_c["cartoes_pro"]
    cantos_f, cartoes_f = dados_v["escanteios_pro"], dados_v["cartoes_pro"]

    exp_gols_casa = (g_pro_c + g_contra_f) / 2
    exp_gols_visitante = (g_pro_f + g_contra_c) / 2
    total_gols_esperados = exp_gols_casa + exp_gols_visitante

    max_gols = 7
    matriz_prob = np.outer(
        poisson.pmf(np.arange(max_gols), exp_gols_casa),
        poisson.pmf(np.arange(max_gols), exp_gols_visitante),
    )

    prob_casa = np.sum(np.tril(matriz_prob, -1)) * 100
    prob_empate = np.sum(np.diag(matriz_prob)) * 100
    prob_visitante = np.sum(np.triu(matriz_prob, 1)) * 100
    prob_btts = np.sum(matriz_prob[1:, 1:]) * 100

    prob_gols_total = np.zeros(max_gols * 2)
    for i in range(max_gols):
      for j in range(max_gols):
        gols_tot = i + j
        prob_gols_total[gols_tot] += matriz_prob[i, j] * 100

    p_menos_1 = prob_gols_total[0]
    p_mais_1 = 100.0 - p_menos_1
    p_menos_2 = np.sum(prob_gols_total[0:2])
    p_mais_2 = 100.0 - p_menos_2
    p_menos_3 = np.sum(prob_gols_total[0:3])
    p_mais_3 = 100.0 - p_menos_3

    # Lógica de Recomendação Inteligente da IA
    melhor_mercado = "Aguardando análise"
    if prob_btts > 58:
      sugestao_ia = (
          "Ambas Marcam (BTTS: Sim)"
          f" ({prob_btts:.1f}% de probabilidade combinada)"
      )
    elif p_mais_2 > 60:
      sugestao_ia = f"Mais de 1.5 / 2 Gols na Partida ({p_mais_2:.1f}%)"
    elif prob_casa > 55:
      sugestao_ia = f"Vitória Simples / Dupla Hipótese: {time_casa}"
    elif prob_visitante > 50:
      sugestao_ia = f"Vitória / Empata Anula: {time_visitante}"
    else:
      sugestao_ia = (
          f"Menos de 2.5 Gols ou Jogo Fechado ({100 - p_mais_2:.1f}%)"
      )

    aba1, aba2, aba3 = st.tabs(
        ["🤖 Palpite da IA & Gols", "📊 Probabilidades 1X2", "💾 Histórico"]
    )

    with aba1:
      st.subheader(f"Confronto: **{time_casa} vs {time_visitante}**")

      # Card de Destaque da IA com Cores Vivas
      st.markdown(
          f"""
                <div class="card-ia">
                    <h3 style="margin: 0; color: #ffffff !important;">🤖 Palpite de Maior Probabilidade (IA)</h3>
                    <p style="font-size: 20px; font-weight: bold; margin: 10px 0 0 0;">👉 {sugestao_ia}</p>
                    <p style="font-size: 13px; margin: 5px 0 0 0; opacity: 0.9;">Média estimada de gols esperados no duelo: <b>{total_gols_esperados:.2f}</b></p>
                </div>
            """,
          unsafe_allow_html=True,
      )

      st.markdown("### 🎯 Linhas Detalhadas de Gols (Over / Under)")
      col_l1, col_l2, col_l3 = st.columns(3)

      with col_l1:
        st.info(
            "**Linha de 1 Gol**\n\n"
            f"• **Mais de 1:** {p_mais_1:.1f}%\n"
            f"• **Menos de 1:** {p_menos_1:.1f}%"
        )
      with col_l2:
        st.info(
            "**Linha de 2 Gols**\n\n"
            f"• **Mais de 2:** {p_mais_2:.1f}%\n"
            f"• **Menos de 2:** {p_menos_2:.1f}%"
        )
      with col_l3:
        st.info(
            "**Linha de 3 Gols**\n\n"
            f"• **Mais de 3:** {p_mais_3:.1f}%\n"
            f"• **Menos de 3:** {p_menos_3:.1f}%"
        )

      st.markdown("#### 📊 Distribuição de Gols Exatos")
      eg1, eg2, eg3, eg4 = st.columns(4)
      eg1.metric("0 Gols (0x0)", f"{prob_gols_total[0]:.1f}%")
      eg2.metric("1 Gol", f"{prob_gols_total[1]:.1f}%")
      eg3.metric("2 Gols", f"{prob_gols_total[2]:.1f}%")
      eg4.metric("3+ Gols", f"{np.sum(prob_gols_total[3:]):.1f}%")

    with aba2:
      st.subheader("📈 Probabilidades de Resultado (1X2) & Outros Mercados")

      c1, c2, c3 = st.columns(3)
      c1.metric(f"Vitória {time_casa}", f"{prob_casa:.1f}%")
      c2.metric("Empate", f"{prob_empate:.1f}%")
      c3.metric(f"Vitória {time_visitante}", f"{prob_visitante:.1f}%")

      st.markdown("---")
      mc1, mc2 = st.columns(2)
      with mc1:
        st.success(f"**🤝 Ambas Marcam (BTTS):** {prob_btts:.1f}%")
        st.success(f"**🚩 Escanteios Totais:** {cantos_c + cantos_f:.1f}")
      with mc2:
        st.success(f"**🟨 Cartões Totais:** {cartoes_c + cartoes_f:.1f}")

      st.markdown("---")
      if st.button("💾 Salvar Este Palpite no Histórico", type="primary"):
        registro = {
            "data": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
            "partida": f"{time_casa} x {time_visitante}",
            "palpite_ia": sugestao_ia,
            "media": round(total_gols_esperados, 2),
        }
        st.session_state["historico_palpites"].append(registro)
        st.success("✅ Palpite salvo com sucesso!")

    with aba3:
      st.subheader("📂 Seus Palpites Guardados")
      if len(st.session_state["historico_palpites"]) == 0:
        st.info("Nenhum palpite guardado ainda.")
      else:
        if st.button("🗑️ Limpar Todos os Registros", type="secondary"):
          st.session_state["historico_palpites"] = []
          st.rerun()

        st.markdown("---")
        for i, p in enumerate(st.session_state["historico_palpites"]):
          st.markdown(
              f"""
                    <div class="card-palpite">
                        <b>[{p['data']}] {p['partida']}</b><br>
                        • Sugestão IA: <b>{p['palpite_ia']}</b><br>
                        • Média Gols: <b>{p['media']}</b>
                    </div>
                """,
              unsafe_allow_html=True,
          )
          if st.button(f"Remover item #{i+1}", key=f"del_{i}"):
            st.session_state["historico_palpites"].pop(i)
            st.rerun()
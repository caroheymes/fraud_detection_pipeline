# src/dashboard/theme.py
"""Module de gestion du thème visuel et de la typographie de l'application Streamlit.

Intègre la police Google Fonts IBM Plex Sans et IBM Plex Mono pour tous les
composants et graphiques.
"""

import plotly.io as pio
import streamlit as st


def apply_theme():
    """Injecte la typographie IBM Plex Sans et IBM Plex Mono dans l'interface Streamlit

    et configure la police par défaut des graphiques Plotly.
    """
    custom_css = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,600&display=swap" rel="stylesheet">
    <style>
        /* Déclarations directes @font-face */
        @font-face {
          font-family: 'IBM Plex Mono';
          font-style: normal;
          font-weight: 400;
          font-display: swap;
          src: url('https://fonts.gstatic.com/s/ibmplexmono/v20/-F63fjptAgt5VM-kVkqdyU8n5ig.ttf') format('truetype');
        }
        @font-face {
          font-family: 'IBM Plex Mono';
          font-style: normal;
          font-weight: 600;
          font-display: swap;
          src: url('https://fonts.gstatic.com/s/ibmplexmono/v20/-F6qfjptAgt5VM-kVkqdyU8n3vAO8lc.ttf') format('truetype');
        }
        @font-face {
          font-family: 'IBM Plex Sans';
          font-style: normal;
          font-weight: 300;
          font-stretch: normal;
          font-display: swap;
          src: url('https://fonts.gstatic.com/s/ibmplexsans/v23/zYXGKVElMYYaJe8bpLHnCwDKr932-G7dytD-Dmu1swZSAXcomDVmadSDtFlzAA.ttf') format('truetype');
        }
        @font-face {
          font-family: 'IBM Plex Sans';
          font-style: normal;
          font-weight: 400;
          font-stretch: normal;
          font-display: swap;
          src: url('https://fonts.gstatic.com/s/ibmplexsans/v23/zYXGKVElMYYaJe8bpLHnCwDKr932-G7dytD-Dmu1swZSAXcomDVmadSD6llzAA.ttf') format('truetype');
        }
        @font-face {
          font-family: 'IBM Plex Sans';
          font-style: normal;
          font-weight: 500;
          font-stretch: normal;
          font-display: swap;
          src: url('https://fonts.gstatic.com/s/ibmplexsans/v23/zYXGKVElMYYaJe8bpLHnCwDKr932-G7dytD-Dmu1swZSAXcomDVmadSD2FlzAA.ttf') format('truetype');
        }
        @font-face {
          font-family: 'IBM Plex Sans';
          font-style: normal;
          font-weight: 600;
          font-stretch: normal;
          font-display: swap;
          src: url('https://fonts.gstatic.com/s/ibmplexsans/v23/zYXGKVElMYYaJe8bpLHnCwDKr932-G7dytD-Dmu1swZSAXcomDVmadSDNF5zAA.ttf') format('truetype');
        }
        @font-face {
          font-family: 'IBM Plex Sans';
          font-style: normal;
          font-weight: 700;
          font-stretch: normal;
          font-display: swap;
          src: url('https://fonts.gstatic.com/s/ibmplexsans/v23/zYXGKVElMYYaJe8bpLHnCwDKr932-G7dytD-Dmu1swZSAXcomDVmadSDDV5zAA.ttf') format('truetype');
        }

        /* Application universelle sur l'application Streamlit et tous ses conteneurs */
        html, body, .stApp, .stApp *, [class*="st-"], [class*="css-"] {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        }

        /* En-tetes et titres */
        h1, h2, h3, h4, h5, h6, [data-testid="stHeadingWithActionElements"], .stTitle {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }

        /* Metriques Streamlit */
        [data-testid="stMetric"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
            font-family: 'IBM Plex Sans', sans-serif !important;
        }

        /* Formulaires, boutons, onglets et navigation */
        .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput, .stTextInput, .stRadio, .stCheckbox, button, .stButton > button, div[data-baseweb="tab-list"] button {
            font-family: 'IBM Plex Sans', sans-serif !important;
        }

        /* Blocs de code et balises monospace */
        code, pre, kbd, samp, .stCodeBlock, [data-testid="stCodeBlock"], [data-testid="stCodeBlock"] * {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        /* Cartes d'accueil et badges */
        .welcome-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 25px;
        }
        .status-badge {
            background-color: #d1fae5;
            color: #065f46;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Harmonisation de la typographie sur les graphiques Plotly
    try:
        if "plotly_white" in pio.templates:
            pio.templates[
                "plotly_white"
            ].layout.font.family = "IBM Plex Sans, sans-serif"
        pio.templates.default = "plotly_white"
    except Exception:
        pass

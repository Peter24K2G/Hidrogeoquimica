import streamlit as st
from PIL import Image
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from wqchartpy import gibbs, triangle_piper, contour_piper, durvo, gaillardet, schoeller, hfed

# Mostrar el logo en la barra lateral
logo = Image.open("images/SIAMSicon.png")  # Ajusta la ruta del logo si es necesario
st.sidebar.image(logo, use_container_width=True)

# Barra lateral con selección de datos
st.sidebar.markdown("### Selección de datos")
data_option = st.sidebar.radio(
    "Elige cómo deseas cargar los datos:",
    ["Ejemplo predeterminado", "Cargar archivo CSV"]
)

# Cargar datos según la selección del usuario
if data_option == "Ejemplo predeterminado":
    # Cargar los datos desde el archivo por defecto
    try:
        df = pd.read_excel('Insumos/CA_SUR.xlsx', sheet_name='Muestreo mgL')
        st.success("Se han cargado los datos predeterminados correctamente.")
    except Exception as e:
        st.error(f"Error al cargar los datos de ejemplo: {e}")
        df = None
else:
    st.write("*Descarga el archivo plantilla*")

    # Cargar el archivo CSV como bytes para la descarga
    try:
        with open("Insumos/PlantillaFQO.csv", "rb") as file:
            edf = file.read()
        st.download_button(label="Descargar CSV", data=edf, file_name="PlantillaFQO.csv", mime="text/csv")
    except Exception as e:
        st.error(f"No se pudo cargar la plantilla: {e}")

    # Permitir al usuario subir su propio archivo CSV
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Archivo CSV cargado correctamente.")
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            df = None
    else:
        df = None

# Definir el número total de páginas
total_pages = 8
if "page" not in st.session_state:
    st.session_state.page = 1

# Funciones de navegación
def next_page():
    if st.session_state.page < total_pages:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

# Mostrar la página actual
st.markdown(f"**Página {st.session_state.page} de {total_pages}**")

# Lógica de cada página
if df is not None:  # Solo ejecutar si hay datos cargados correctamente
    if st.session_state.page == 1:
        st.markdown("""## Introducción a los diagramas hidrogeoquímicos.
        Datos cargados:
        """)
        st.write(df)

    elif st.session_state.page == 2:
        st.markdown("## Diagrama de Gibbs")
        st.markdown("Descripción del diagrama de Gibbs...")

        if st.button("Generar diagrama de Gibbs"):
            df['Label'] = ""
            gibbs.plot(df, unit='mg/L', figname='images/Gibbs_diagram', figformat='jpg')
            st.image("images/Gibbs_diagram.jpg", caption="Diagrama de Gibbs", use_container_width=True)

    elif st.session_state.page == 3:
        st.markdown("## Diagrama de Piper")
        st.markdown("Descripción del diagrama de Piper...")

        option = st.selectbox("Selecciona el gráfico:", ["Diagrama de Piper", "Diagrama de contornos de Piper"])

        if option == "Diagrama de Piper":
            df['Label'] = ""
            triangle_piper.plot(df, unit='mg/L', figname='images/Piper_diagram', figformat='jpg')
            st.image("images/Piper_diagram.jpg", caption="Diagrama de Piper", use_container_width=True)

        elif option == "Diagrama de contornos de Piper":
            df['Label'] = ""
            contour_piper.plot(df, unit='mg/L', figname='images/Contour_Piper', figformat='jpg')
            st.image("images/Contour_Piper.jpg", caption="Diagrama de contornos de Piper", use_container_width=True)

    elif st.session_state.page == 4:
        st.markdown("## Diagrama de Durov")

        if st.button("Generar diagrama de Durov"):
            df['Label'] = ""
            durvo.plot(df, unit='mg/L', figname='images/Durov_diagram', figformat='jpg')
            st.image("images/Durov_diagram.jpg", caption="Diagrama de Durov", use_container_width=True)

    elif st.session_state.page == 5:
        st.markdown("## Diagrama de Gaillardet")

        if st.button("Generar diagrama de Gaillardet"):
            df['Label'] = ""
            gaillardet.plot(df, unit='mg/L', figname='images/Gaillardet_diagram', figformat='jpg')
            st.image("images/Gaillardet_diagram.jpg", caption="Diagrama de Gaillardet", use_container_width=True)

    elif st.session_state.page == 6:
        st.markdown("## Diagrama de Schoeller")

        if st.button("Generar diagrama de Schoeller"):
            df['Label'] = ""
            schoeller.plot(df, unit='mg/L', figname='images/Schoeller_diagram', figformat='jpg')
            st.image("images/Schoeller_diagram.jpg", caption="Diagrama de Schoeller", use_container_width=True)

    elif st.session_state.page == 7:
        st.markdown("## Relaciones químicas en el agua")
        st.markdown("Análisis de relaciones químicas con gráficos de dispersión interactivos.")

        fig = px.scatter(df, x="HCO3", y="Ca+Mg", color="Cluster",
                         title="Relación Ca + Mg vs HCO3",
                         labels={"HCO3": "HCO3 (meq/L)", "Ca+Mg": "Ca + Mg (meq/L)"})
        st.plotly_chart(fig)

    elif st.session_state.page == 8:
        st.markdown("## Diagrama HFE-D")

        if st.button("Generar diagrama HFE-D"):
            df['Label'] = ""
            hfed.plot(df, unit='mg/L', figname='images/HFE-D_diagram', figformat='jpg')
            st.image("images/HFE-D_diagram.jpg", caption="Diagrama HFE-D", use_container_width=True)

# Navegación entre páginas
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.button("Anterior", on_click=prev_page)
with col3:
    st.button("Siguiente", on_click=next_page)

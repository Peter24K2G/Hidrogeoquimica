import streamlit as st
from PIL import Image
import pandas as pd
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

from wqchartpy import gibbs
from wqchartpy import triangle_piper
from wqchartpy import contour_piper
from wqchartpy import durvo
from wqchartpy import gaillardet
from wqchartpy import schoeller



# Mostrar el logo en la barra lateral
logo = Image.open("images\MEGIAicon.png")  # Asegúrate de colocar la ruta correcta de tu logo
st.sidebar.image(logo, use_container_width=True)

# Barra lateral con título e información
with st.sidebar:
    st.markdown("""
        # Hidrogeoquímica
        Diplomado MEGIA
    """)
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # Ajusta el número de <br> según necesites

    st.sidebar.image("images/Profile.png", width=80)
    # Texto en la barra lateral
    st.sidebar.markdown(
        """
        <div style="text-align: left;">
            <p><strong>Prof. Adriana Piña</strong><br> Universidad Nacional de Colombia</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""# Análisis Estadístico e Hidrogeoquímico

## Introducción
El análisis de datos es una herramienta fundamental para comprender la variabilidad y distribución de las muestras hidrogeoquímicas. Para ello, se emplean métodos estadísticos y representaciones gráficas que permiten evaluar la composición química del agua y sus procesos asociados.

## Análisis Estadístico
El análisis estadístico permite caracterizar la calidad del agua a partir de diferentes pruebas, tales como:

- **Análisis descriptivo:** Determina medidas como la media, mediana, máximos, mínimos y desviaciones estándar para evaluar la dispersión y tendencia central de los datos.
- **Diagramas de caja y bigotes:** Visualizan la distribución de los datos e identifican valores atípicos.
- **Pruebas de normalidad:** Verifican si los datos siguen una distribución normal mediante pruebas como Shapiro-Wilk o Kolmogorov-Smirnov.
- **Análisis de varianza (ANOVA) y pruebas no paramétricas:** Comparan diferencias significativas entre grupos de datos.

## Diagramas Hidrogeoquímicos
Para interpretar la composición del agua subterránea y su evolución geoquímica, se utilizan diagramas especializados:

- **Diagrama de Gibbs:** Evalúa el origen del agua según procesos de precipitación, evaporación o interacción roca-agua.
- **Diagrama de Piper:** Permite clasificar los tipos de agua según sus cationes y aniones dominantes.
- **Diagrama de Durov:** Representa gráficamente la evolución química del agua.
- **Diagrama de Gaillardet:** Relaciona la composición química del agua con su interacción con distintos tipos de rocas.
- **Diagrama de Schoeller:** Comparación de múltiples muestras en una escala logarítmica.
- **Diagrama HFE-D:** Analiza la influencia de la evaporación y el intercambio iónico en la composición del agua.
- **Diagrama de Stiff:** Visualiza patrones característicos de la química del agua mediante polígonos.

## Aplicaciones
Estos métodos y diagramas son ampliamente utilizados en estudios de hidrogeología, monitoreo de calidad del agua y evaluación de la interacción agua-roca en acuíferos.

---""")

# df = pd.read_excel('Insumos\BDD_MEGIA.xlsx')
# st.write(df.head())

# df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
# df["LONG"] = pd.to_numeric(df["LONG"], errors="coerce")


# # Crear el mapa centrado en las coordenadas medias del DataFrame
# m = folium.Map(location=[df["LAT"].mean(), df["LONG"].mean()], zoom_start=8)

# # Agregar capa de control de mapa (ej. CartoDB positron)
# folium.LayerControl().add_to(m)

# # Agregar marcador con MarkerCluster para agrupar los puntos cercanos
# marker_cluster = MarkerCluster(name="Puntos de monitoreo").add_to(m)

# # Calcular los límites del mapa para ajustarlo a los puntos de monitoreo
# bounds = [
#     [df["LAT"].min(), df["LONG"].min()],
#     [df["LAT"].max(), df["LONG"].max()]
# ]
# m.fit_bounds(bounds)

# # Función para agregar marcadores con popup con información adicional
# def plot_station(row):
#     html = row.to_frame("Información").to_html(classes="table table-striped table-hover table-condensed table-responsive")
#     popup = folium.Popup(html, max_width=1000)
#     folium.Marker(location=[row["LAT"], row["LONG"]], popup=popup).add_to(marker_cluster)

# # Aplicar la función para agregar los marcadores
# df.apply(plot_station, axis=1)

# # Mostrar el mapa en Streamlit
# st.markdown("### Visualización de las ubicaciones de los datos")
# folium_static(m)  # Renderiza el mapa en Streamlit



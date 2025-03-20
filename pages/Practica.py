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
        medf1 = df[['Sample','pH', 'HCO3', 'CO3', 'Ca', 'Mg', 'K', 'Na', 'Cl', 'SO4', 'TDS']]
        medf2 = medf1.dropna()
        iris1 = medf2[['pH', 'HCO3', 'CO3', 'Ca', 'Mg', 'K', 'Na', 'Cl', 'SO4', 'TDS']]
        standardisedX = scale(iris1)
        standardisedX = pd.DataFrame(standardisedX, index=iris1.index, columns=iris1.columns)
    
        standardisedX.apply(np.mean)
        standardisedX.apply(np.std)
        pca = PCA().fit(standardisedX)
    
        def pca_summary(pca, standardised_data, out=True):
            names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
            a = list(np.std(pca.transform(standardised_data), axis=0))
            b = list(pca.explained_variance_ratio_)
            c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
            columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
            summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
            return summary
        summary = pca_summary(pca, standardisedX)
        st.write(summary)
        st.markdown("""
        ### Varianza acumulada
        Se puede tambien verificar el porcentaje acumulado de la varianza explicada por los componentes. Es importante observar que con sólo 2 componentes se alcanza un 64% y con el tercer componente hasta el 80%.
        """)
        st.markdown("""
        ## Selección de número de componentes principales a retener
        ### Scree plot
        Se pueden retener los componentes de acuerdo con el cambio de pendiente más fuerte en el Scree plot (hasta el componente 4). 
        """)
    
        # Función corregida
        def screeplot(pca, standardised_values):
            y = np.std(pca.transform(standardised_values), axis=0)**2
            x = np.arange(len(y)) + 1
    
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(x, y, "o-", markersize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(["Comp."+str(i) for i in x], rotation=60)
            ax.set_ylabel("Varianza Explicada")
            ax.set_xlabel("Componentes Principales")
            ax.set_title("Scree Plot (Varianza Explicada por Componente)")
            
            return fig  # Devuelve la figura correctamente
    
        # Mostrar en Streamlit
        st.pyplot(screeplot(pca, standardisedX))
    
        st.markdown("""
        ### Citerio de Kaisser
        Se pueden retener los componentes cuya varianza este por encima de 1. Para el ejemplo se conservarían los tres primeros componentes.
        """)
        st.write(summary.sdev**2)
    
        #Asignar índices a la base de datos.
        label = iris1.columns
        standardisedX.set_index(medf['Sample'], inplace=True) 
        # model = pcaf(n_components=0.95)
        model = pcaf(n_components=6) #Solo conserva 3 componentes
        results = model.fit_transform(standardisedX)
        fig, ax = model.biplot(n_feat=9, PC=[0,1])
    
        st.write("# Peso de los componentes principales")
        st.pyplot(fig)
    
        st.divider()
    
        st.write("# HHCA (Hierarchical Cluster Analysis)")
        # Enlace jerárquico
        enlace = linkage(standardisedX, 'ward')
    
        # Dendrograma
        dendrograma = dendrogram(enlace)
    
        def plot_dendrogram():
            fig, ax = plt.subplots(figsize=(8, 5))
            dendrogram(enlace, ax=ax)
            ax.set_title("Dendrograma Jerárquico")
            ax.set_ylabel("Distancia")
            ax.set_xlabel("Muestras")
            return fig  # Retornar la figura
    
        # Mostrar el dendrograma en Streamlit
        st.pyplot(plot_dendrogram())

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

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
import os


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from pca import pca as pcaf

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pingouin as pg
from pingouin import kruskal
from scipy.stats import kstest, norm, f_oneway
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# from wqchartpy import gibbs
# from wqchartpy import triangle_piper
# from wqchartpy import contour_piper
# from wqchartpy import durvo
# from wqchartpy import gaillardet
# from wqchartpy import schoeller
# from wqchartpy import hfed
# from wqchartpy import stiff


# Mostrar el logo en la barra lateral
logo = Image.open("images/SIAMSicon.png")  # Asegúrate de colocar la ruta correcta de tu logo
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

# Definir el número total de páginas
total_pages = 16  # Cambia este número según el número de páginas que necesites
if "page" not in st.session_state:
    st.session_state.page = 1

# Funciones para navegar entre páginas
def next_page():
    if st.session_state.page < total_pages:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

def reset_page():
    st.session_state.page = 1  # Reiniciar la página a 1

# Mostrar el breadcrumb
breadcrumb = f"Página {st.session_state.page} de {total_pages}"
st.markdown(f"{breadcrumb}")

#----------------------------------------------------------------------------------------------
if st.session_state.page == 1:
#----------------------------------------------------------------------------------------------
    st.markdown("""
    # Análisis Estadístico
     
    El análisis estadístico realizado tiene como objetivo evaluar y comprender la distribución y variabilidad de los datos, para luego determinar las pruebas estadísticas más adecuadas. A continuación, se detallan los pasos realizados durante el análisis y las pruebas estadísticas empleadas.
                
    ## Introducción

    Se llevó a cabo un análisis estadístico preliminar de los datos con el fin de obtener una visión general sobre la variabilidad y distribución de los mismos. En primer lugar, se calcularon estadísticas descriptivas, como los valores promedio, máximos, mínimos y desviaciones estándar, para tener una primera aproximación sobre la dispersión y tendencia central de los datos. Posteriormente, se utilizó un diagrama de caja y bigotes (boxplot) para visualizar la distribución de los datos, detectar posibles valores atípicos, y evaluar la simetría de los grupos.

    Una vez obtenida esta información inicial, se procedió a analizar la normalidad de los datos, ya que este análisis es crucial para elegir las pruebas estadísticas adecuadas. Dependiendo de si los datos siguen una distribución normal, se aplicó la prueba estadística correspondiente, como ANOVA o Kruskal-Wallis, para comparar las medias de los diferentes grupos. Finalmente, se realizó una prueba post-hoc de Dunn-Bonferroni para identificar diferencias significativas entre las medias de los grupos.
    """)

    st.markdown("# Datos de ejemplo")

    df = pd.read_csv("Insumos/PlantillaFQO.csv")
    df
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 2:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        ## Prueba 1: Análisis Descriptivo

        ### Descripción:
        El análisis descriptivo incluye el cálculo de valores clave como el **promedio**, **máximo**, **mínimo** y **desviación estándar**. Estos valores proporcionan una visión general sobre la tendencia central y la dispersión de los datos.

        - **Promedio**: Es el valor central de los datos y proporciona una indicación de la tendencia general.
        - **Máximo y Mínimo**: Ayudan a entender el rango de los datos y si existen valores extremos que podrían influir en el análisis.
        - **Desviación Estándar**: Mide la dispersión de los datos en relación con el promedio. Una desviación estándar alta indica que los datos están más dispersos, mientras que una desviación estándar baja indica que los datos son más consistentes.

        Este paso inicial es crucial para tener una visión preliminar de los datos y entender su variabilidad antes de proceder con pruebas estadísticas más complejas.
    """)

    df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # Seleccionar las columnas relevantes
    columns_of_interest = ["pH", "HCO3", "Ca", "Mg", "K", "Na", "Cl", "SO4", "TDS"]
    # Generar estadísticas descriptivas
    desc_stats = df[columns_of_interest].describe().transpose()
    # Crear una nueva columna para el "Número de muestras"
    desc_stats["# de muestras"] = df[columns_of_interest].count()
    # Añadir las unidades para cada parámetro
    desc_stats["Unidades"] = ["Und. De pH", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L"]
    # Reorganizar las columnas para que coincidan con la estructura de la tabla
    desc_stats = desc_stats[["Unidades", "# de muestras", "mean", "std", "max", "min"]]
    # Renombrar las columnas para que coincidan con la imagen
    desc_stats.columns = ["Unidades", "# de muestras", "Promedio", "Desviación estándar", "Máximo", "Mínimo"]
    desc_stats[["Promedio", "Desviación estándar", "Máximo", "Mínimo"]] = desc_stats[["Promedio", "Desviación estándar", "Máximo", "Mínimo"]].round(2)
    # Mostrar el resultado en Streamlit
    st.write(desc_stats)
    
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 3:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        ## Prueba 2: Diagrama de Caja y Bigotes (Boxplot)

        ### Descripción:
        Un **diagrama de caja y bigotes** (boxplot) es una herramienta visual que permite observar la distribución de los datos. Este gráfico muestra:

        - **Mediana**: Representa el valor central de los datos.
        - **Cuartiles**: Indican cómo se distribuyen los datos. El primer cuartil (Q1) es el valor debajo del cual se encuentra el 25% de los datos, y el tercer cuartil (Q3) es el valor debajo del cual se encuentra el 75% de los datos.
        - **Valores Atípicos**: Los puntos que se encuentran fuera de los bigotes del boxplot se consideran valores atípicos o **outliers**. Estos valores son importantes porque pueden indicar errores de medición o comportamientos inusuales en los datos.

        ### Objetivo:
        El boxplot permite detectar **dispersión**, **asimetría** y **valores atípicos**, lo que facilita la comprensión visual de cómo se distribuyen los datos.
    """)
    mgdf = pd.read_excel('Insumos/CA_SUR.xlsx',sheet_name='Muestreo mgL')
    # Filtrar datos según la clase deseada
    clase_deseada = 'ZEM VI'
    df_fil = mgdf[mgdf['Campo'] == clase_deseada] if clase_deseada else mgdf

    # Variables a analizar
    iones = ['Mg', 'Na', 'Ca', 'K', 'HCO3', 'Cl', 'SO4', 'NO3']
    iones_superindice = {
        'Mg': 'Mg²⁺',
        'Na': 'Na⁺',
        'Ca': 'Ca²⁺',
        'K': 'K⁺',
        'HCO3': 'HCO₃⁻',
        'Cl': 'Cl⁻',
        'SO4': 'SO₄²⁻',
        'NO3': 'NO₃⁻'
    }

    # Reorganizar el DataFrame
    df_melted = df_fil.melt(id_vars=['Marker'], value_vars=iones, var_name='Variable', value_name='Valor')
    df_melted['Variable'] = df_melted['Variable'].replace(iones_superindice)

    # Crear el boxplot con Matplotlib
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x='Variable', y='Valor', palette='Set3', showmeans=True)

    # Personalizar el gráfico
    plt.yscale('log')  # Escala logarítmica
    #plt.ylim(bottom=-1)  # Límite inferior en 0
    #plt.title('Boxplot de iones mayoritarios ZEM NORTE', fontsize=16, fontfamily='Times New Roman')
    plt.xlabel('Parametro', fontsize=20, fontfamily='Times New Roman')
    plt.ylabel('Concentración (mg/L)', fontsize=20, fontfamily='Times New Roman')
    plt.xticks(fontsize=20, fontfamily='Times New Roman')
    plt.yticks(fontsize=20, fontfamily='Times New Roman')


    # Crear elementos de leyenda personalizados
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', label='Rango intercuartílico'),
        Line2D([0], [0], color='black', label='Mediana', linestyle='-'),
        Line2D([0], [0], color='green', label='Media', marker='^', linestyle='', markersize=8),
        Line2D([0], [0], color='black', marker='o', markerfacecolor='white', label='Datos atípicos', linestyle='', markersize=8),
    ]

    plt.legend(handles=legend_elements, loc='best', fontsize=12, title='Leyenda', title_fontsize=12, frameon=True)

    # Mostrar el gráfico
    plt.tight_layout()
    st.pyplot(fig)
      
#----------------------------------------------------------------------------------------------

elif st.session_state.page == 4:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        ## Prueba 3: Análisis de Normalidad

        ### Descripción:
        Antes de aplicar cualquier prueba estadística, es fundamental determinar si los datos siguen una **distribución normal**, ya que muchas pruebas paramétricas requieren que los datos sean normales para ser válidas. El análisis de normalidad puede realizarse de diversas maneras, entre las que se incluyen:

        - **Pruebas estadísticas** como la **Prueba de Shapiro-Wilk** o la **Prueba de Kolmogórov-Smirnov**.
        - **Visualización** a través de gráficos como el **histograma** o el **diagrama Q-Q** (quantile-quantile), que permiten comparar la distribución de los datos con la distribución normal.

        ### Objetivo:
        Este análisis es crucial para determinar qué pruebas estadísticas son apropiadas. Si los datos siguen una distribución normal, se puede aplicar una prueba paramétrica como **ANOVA**. Si no siguen una distribución normal, se recurrirá a una prueba no paramétrica, como **Kruskal-Wallis**.

    """)
    medf = pd.read_excel('Insumos/CA_SUR.xlsx',sheet_name='Muestreo meqL')
    col1, col2 = st.columns(2)
    with col1:
        group = st.selectbox("Seleccione la agrupacion en la que va a hacer el análisis de normalidad:",["Tipo","Clase","Campo","Season"])
    with col2:
        parameter = st.selectbox("Seleccione el parámetro al cuál se le va a hacer el análisis de normalidad:",
                             ['Mg', 'Na', 'Ca', 'K', 'HCO3', 'Cl', 'SO4', 'NO3'])
    normality_result = pg.normality(data=medf, dv=parameter, group=group)  # Cambia 'Clase' por la columna de agrupación adecuada
    st.write(normality_result) 

    st.markdown("""
        ## Prueba 4: ANOVA (Análisis de Varianza)

        ### Descripción:
        **ANOVA** (Análisis de Varianza) es una prueba estadística que se utiliza para comparar las medias de tres o más grupos. El objetivo de ANOVA es determinar si existen diferencias significativas entre las medias de los grupos. Esta prueba asume que los datos siguen una distribución normal y que las varianzas entre los grupos son homogéneas (es decir, las varianzas de los grupos no son significativamente diferentes).

        #### ¿Cómo funciona?
        - Se calcula la varianza dentro de los grupos y la varianza entre los grupos.
        - Si la varianza entre los grupos es mucho mayor que la varianza dentro de los grupos, se concluye que hay diferencias significativas entre los grupos.

        ### Objetivo:
        ANOVA se utiliza cuando los datos son **normales** y se busca comparar las medias de más de dos grupos.
    """)
    grupo_col = group  # Cambia por el nombre real de tu columna de grupos
    valor_col = parameter  # Cambia por el nombre real de tu columna de valores

    # Agrupar los datos por los grupos y preparar listas para el ANOVA
    grupos = medf[grupo_col].unique()
    datos_grupos = [medf[medf[grupo_col] == g][valor_col].values for g in grupos]

    # Realizar el ANOVA de una vía
    anova_resultado = f_oneway(*datos_grupos)

    # Mostrar resultados
    st.write("# Resultados del ANOVA:")
    st.write(f"Los datos agrupados por **{group}** para el parámetro **{parameter}**")
    st.write(f"F-statistic: {anova_resultado.statistic:0.3f}")
    st.write(f"P-value: {anova_resultado.pvalue:0.3f}")

    # Interpretación
    alpha = 0.05  # Nivel de significancia
    if anova_resultado.pvalue < alpha:
        st.write("\nEl resultado es significativo: Hay diferencias entre los grupos.")
    else:
        st.write("\nEl resultado no es significativo: No hay diferencias entre los grupos.")
# Mostrar el contenido según la página actual

#----------------------------------------------------------------------------------------------
if st.session_state.page == 5:
#----------------------------------------------------------------------------------------------
    st.markdown("""
    # ANÁLISIS ESTADÍSTICO MULTIVARIADO
    Se presentan algunas rutinas básicas para el Análisis de Componentes Principales y de Agrupamiento (Cluster) utilizando resultados de análisis fisicoquímicos de agua subterránea (CAMPUS UNAL 2023). Las concentraciones se reportan en meq/L.
    Para la preparación de la base de datos, se generan algunos estadísticos y gráficos descriptivos, evaluación de la normalidad de los parámetros y estandarización de los datos.
    """)
    medf = pd.read_csv('Insumos/PlantillaFQO.csv')

    medf1 = medf[['Sample','pH', 'HCO3', 'CO3', 'Ca', 'Mg', 'K', 'Na', 'Cl', 'SO4', 'TDS']]

    medf2 = medf1.dropna()

    st.markdown("""
    ##  Matriz de correlación de los parámetros
    Se desea realizar una correlación entre los paramétros medidos, para ello se obtiene una matriz de correlación entre los componentes de las muestras, con 1 en la diagonal principal que correlaciona la misma variable. 
    """)

    iris = medf2[['pH', 'HCO3', 'CO3', 'Ca', 'Mg', 'K', 'Na', 'Cl', 'SO4', 'TDS']]
    # iris = iris.drop('CO3',axis=1)
    matriz_pearson = iris.corr(method='pearson')

    #Figura de matriz de correlación
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_pearson, annot=True, fmt=".1f",cmap='coolwarm', linewidths=.5)
    plt.title('Matriz de Correlación de Pearson')
    st.pyplot(fig)
#----------------------------------------------------------------------------------------------
if st.session_state.page == 5:
#----------------------------------------------------------------------------------------------
    st.markdown("# Analisis de componentes principales")
    medf = pd.read_csv('Insumos/PlantillaFQO.csv')
    medf1 = medf[['Sample','pH', 'HCO3', 'CO3', 'Ca', 'Mg', 'K', 'Na', 'Cl', 'SO4', 'TDS']]
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
    st.write("Pesos:", results['loadings'])
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



#----------------------------------------------------------------------------------------------
if st.session_state.page == 6:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        # Introducción al Análisis Hidrogeoquímico

        El análisis hidrogeoquímico permite estudiar la composición iónica de las aguas subterráneas y su interacción con las formaciones geológicas. Un aspecto clave es el **balance iónico**, que evalúa la relación entre los cationes y aniones presentes, ayudando a identificar procesos como la disolución de carbonatos y sulfatos o la intrusión salina. Además de los diagramas hidrogeoquímicos, como los diagramas de Gibbs, Piper y Stiff, que visualizan las relaciones de iones y la mineralización del agua, este análisis también permite detectar la influencia de fenómenos geológicos y químicos sobre la calidad del agua.                
""")
#----------------------------------------------------------------------------------------------
# Mostrar el contenido según la página actual
#----------------------------------------------------------------------------------------------
if st.session_state.page == 7:
#----------------------------------------------------------------------------------------------
    st.markdown("""
    # Introducciona a los diagramas hidrogeoquimicos
    ### Balance ionico

    ## ¿Qué es el Balance Iónico?
    El balance iónico es un método utilizado en hidrogeoquímica para evaluar la calidad de los análisis químicos del agua, verificando la consistencia entre las concentraciones de cationes y aniones disueltos.

    ## Cálculo del Balance Iónico
    Se basa en la comparación de la suma de cargas de los **cationes** $( Ca^{2+}, Mg^{2+}, Na^+, K^+ )$) y los **aniones** ($( HCO_3^-, SO_4^{2-}, Cl^- $) en **meq/L** (miliequivalentes por litro), mediante la siguiente ecuación:

    Balance Iónico (\%) = $ ((\sum {Cationes} - \sum {Aniones})/(\sum {Cationes} + \sum {Aniones})) * 100 $

    ## Interpretación
    - Un **balance cercano a 0%** indica que los análisis químicos son confiables.
    - Valores **mayores a ±5%** pueden sugerir errores en la medición, contaminación de muestras o presencia de especies no consideradas en el análisis.
    - El balance iónico es esencial para garantizar la validez de los datos en estudios hidroquímicos.

    """)
    df = pd.read_csv('Insumos/PlantillaFQO.csv')
    
    cationes = ['Ca', 'Mg', 'Na', 'K']
    aniones = ['HCO3', 'SO4', 'Cl']

    # Sumar cationes y aniones por fila
    df['Suma Cationes'] = df[cationes].sum(axis=1)
    df['Suma Aniones'] = df[aniones].sum(axis=1)

    # Calcular el balance iónico en porcentaje
    df['Balance Ionico (%)'] = ((df['Suma Cationes'] - df['Suma Aniones']) / 
                                (df['Suma Cationes'] + df['Suma Aniones'])) * 100

    # Seleccionar solo las columnas de interés
    tabla_balance = df[['Sample','Suma Cationes', 'Suma Aniones', 'Balance Ionico (%)']]

    # Mostrar la tabla resultante
    st.write(tabla_balance)
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 8:
#---------------------------------------------------------------------------------------------
    st.markdown("""
        ## Diagrama de Gibbs
        El diagrama de Gibbs (1970) relaciona los sólidos disuletos totales como función de la dominanción de $Na^+$ o $Ca^{2+}$ y $Cl^-$ o $HCO_3^-$, ilustrando los procesos de calidad de agua en una cáscara de nuez. A bajas concentraciones disueltas, el agua lluvia sin mucha reacción geoquímica contribuya como iones dominantes son $Na^+$ y $Cl^-$. Cuando entra en contacto con calcita y silicatos de calcio de rápida disolución, el agua aumenta su contenido relativo de $Ca^{2+}$ y $HCO_3^-$. Si hay evaporación concentra la solución, el $Ca^{2+}$ y el $HCO_3^-$ se pierden por precipitación de $CaCO_3$ y la composición del agua cambia hacia arriba en la figura y vuelve a la composición del mar dominada por $Na^-$ y $Cl^-$.
    """)
    # df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # gibbs.plot(df, unit='mg/L', figname='imagesTeoricas/Gibbs diagram', figformat='jpg')
    st.image("imagesTeoricas/Gibbs diagram.jpg", caption="Diagrama de Gibbs", use_container_width=True)    
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 9:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        ## Diagrama de Piper
        El diagrama de Piper fue introducido en 1944 y ha sido desde entonces una herramienta fundamental para el análisis hidrogeoquímico para la clasificación del agua, la determinación del potencial de mezcla y la identificación de las reacciones químicas que controlan determinado sistema.

        Consiste en un diagrama con dos campos triangulares que representan los aniones ($SO_4^{2+}$ y $Cl^-$) y los cationes ($Ca^{2+}$ y $Mg^{2+}$) por separado, y $Na^+$ y $K^+$ y la alcalinidad agrupados. Una vez ubicados los cationes y los aniones en cada triángulo, los puntos se proyectan hacia el rombo y donde se intersecta la proyección de los cationes y los aniones. Este punto permite interpretar el tipo de agua.

        A continuación se presentan las características químicas del agua subterránea en las diferentes zonas del diagrama de Piper:

    """)
    # Cargar las imágenes usando st.image()
    st.image("Fig_Aux/Pipper_expl.jpg", caption="Diagrama de Piper", use_container_width=True)
    st.markdown("""
        y los tipos o facies del agua sobre el diagrama:
    """)
    st.image("Fig_Aux/Pipper_expl_2.jpg", caption="Tipos de Facies de Agua", use_container_width=True)
    st.markdown("""
        o resumidos en una tabla:
    """)
    st.image("Fig_Aux/Facies.JPG", caption="Facies en el Diagrama de Piper", use_container_width=True)
    st.markdown("""
        Recientemente, algunos autores han propuesto la inclusión de técnicas estadísticas para su construcción y mejora, buscando visualizar información adicional como las relaciones $Ca^{2+}$/$Mg^{2+}$ vs $Cl^{-}$/$SO_4^{2}$, útil para identificar condiciones de intercambio catiónico.
    """)
    st.image("Fig_Aux/ILR_Piper.jpg", caption="ILR-Piper", use_container_width=True)
    st.markdown("""
        Tomado de: Shelton et al., 2018.
    """)
    # df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # triangle_piper.plot(df, unit='mg/L', figname='imagesTeoricas/triangle Piper diagram', figformat='jpg')
    # Mostrar la imagen generada
    st.image("imagesTeoricas/triangle Piper diagram.jpg", caption="Diagrama de Piper", use_container_width=True)

    st.divider()

    st.markdown("### Diagrama de contornos de Piper")
    # contour_piper.plot(df, unit='mg/L', figname='imagesTeoricas/contour-filled Piper diagram', figformat='jpg')
    st.image("imagesTeoricas/contour-filled Piper diagram.jpg", caption="Diagrama de contornos de Piper", use_container_width=True)
      
#----------------------------------------------------------------------------------------------

elif st.session_state.page == 10:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        ## Diagrama de Durov
        Es un digrama tri-linear donde se grafican las concentraciones de los cationes y aniones mayoritarios en meq/L en dos triangulos separados. Las concentrciones en cada triangulo se proyectan sobre el cuadrado central, el cual representa el caracter químico general de la muestra. También es posible graficar otra caracteristica química como los Sólidos Disueltos Totales, el pH, la conductividad, etc.
    """)

    # df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # # Botón para generar el gráfico
    # durvo.plot(df, unit='mg/L', figname='imagesTeoricas/Durvo diagram', figformat='jpg')
    # Mostrar la imagen generada
    st.image("imagesTeoricas/Durvo diagram.jpg", caption="Diagrama de Durov", use_container_width=True)   
#----------------------------------------------------------------------------------------------

elif st.session_state.page == 11:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        # Diagrama de Gaillardet
        Grafica la relación $Ca^{2+}$/$Na^{+}$ vs $HCO_3^{-}$/$Na^{+}$ y $Ca^{2+}$/$Na^{+}$ vs $Mg^{2+}$/$Na^{+}$, identificando su relación con rocas evaporitas, silicatos o carbonatos.
    """)
    # df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # # Llamar a la función para generar el gráfico
    # gaillardet.plot(df, unit='mg/L', figname='imagesTeoricas/Gaillardet diagram', figformat='jpg')        # Mostrar la imagen generada
    st.image("imagesTeoricas/Gaillardet diagram.jpg", caption="Diagrama de Gaillardet", use_container_width=True)
#----------------------------------------------------------------------------------------------

elif st.session_state.page == 12:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        # Diagrama de Schoeller
        Es in grafico en escala semilogaritmica, en donde la abscisa (escala aritmetica) se organizan los iones y cationes a distancias equidistantes. Los puntos son unidos con líneas rectas. Dentro de las ventajas de estos diagramas es que se pueden graficar de manera simulatánea diferentes muestras lo que permite su comparación.
    """)

    # df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # schoeller.plot(df, unit='mg/L', figname='imagesTeoricas/Schoeller diagram', figformat='jpg')        # Mostrar la imagen generada
    st.image("imagesTeoricas/Schoeller diagram.jpg", caption="Diagrama de Schoeller", use_container_width=True)
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 13:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        # Diagrama de Stiff
        En este gráfico, los datos analíticos se grafican en cuatro ejes horizontales paralelos equdistantes uno del otro. Estos diagramas tienen la desventaja que son individuales por muestra. Sin embargo, son de gran utilidad para visualizar las diferencias en la distribución de los aniones y los cationes basado en sus patrones.
    """ )
    # df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # stiff.plot(df, unit='mg/L', figname='imagesTeoricas/Stiff diagram', figformat='jpg')
    files = [x for x in os.listdir("imagesTeoricas") if x.startswith("Stiff")]
    imageselection = st.selectbox("Imagen ejemplo a visualizar:",
                    files)
    st.image(f"imagesTeoricas\{imageselection}", caption="Diagrama de HFE-D", use_container_width=True)
    #----------------------------------------------------------------------------------------------
elif st.session_state.page == 14:
#----------------------------------------------------------------------------------------------
    st.markdown("""
    # Diagrama HFE-D
    El **diagrama HFE-D** es una herramienta hidrogeoquímica utilizada para clasificar aguas subterráneas según su composición iónica. Representa la relación entre cationes y aniones, ayudando a identificar el tipo de agua y los procesos geológicos que afectan su calidad, como la disolución de carbonatos o evaporitas.
    """ )
    # df = pd.read_csv('Insumos/PlantillaFQO.csv')
    # hfed.plot(df, unit='mg/L', figname='imagesTeoricas/HFE-D diagram', figformat='jpg')
    st.image("imagesTeoricas/HFE-D diagram.jpg", caption="Diagrama de HFE-D", use_container_width=True)
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 15:
#----------------------------------------------------------------------------------------------
       # Cargar los datos
    df = pd.read_csv('Insumos/PlantillaFQO.csv')

    # Convertir 'Cluster' a tipo texto (string)
    df["Label"] = df["Label"].astype(str)

    # Sumar las concentraciones de Ca y Mg
    df["Ca+Mg"] = df["Ca"] + df["Mg"]

    # Crear gráfico interactivo de dispersión de Ca + Mg vs HCO3 con colores discretos
    fig = px.scatter(df, x="HCO3", y="Ca+Mg", color="Label", 
                    labels={"HCO3": "HCO3 (meq/L)", "Ca+Mg": "Ca + Mg (meq/L)"},
                    title="Relación Ca + Mg vs HCO3 en meq/L", 
                    hover_data=["Sample","Ca+Mg", "HCO3", "Label"],
                    color_discrete_sequence=px.colors.qualitative.Set1)  # Barra de color discreta

    # Agregar la línea 1:1
    min_val = min(df["Ca+Mg"].min(), df["HCO3"].min())
    max_val = max(df["Ca+Mg"].max(), df["HCO3"].max())

    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', line=dict(color='red', dash='dash'), name="Relación 1:1"))

    # Mostrar gráfico interactivo

    # Sumar las concentraciones de Na y K
    df["Na+K"] = df["Na"] + df["K"]

    # Crear gráfico interactivo de dispersión de Na + K vs Cl con colores discretos
    fig1 = px.scatter(df, x="Cl", y="Na+K", color="Label", 
                    labels={"Cl": "Cl (meq/L)", "Na_K_meq": "Na + K (meq/L)"},
                    title="Relación Na + K vs Cl en meq/L", 
                    hover_data=["Sample","Na+K", "K", "Label"],
                    color_discrete_sequence=px.colors.qualitative.Set1)  # Barra de color discreta

    # Agregar la línea 1:1
    min_val = min(df["Na+K"].min(), df["Cl"].min())
    max_val = max(df["Na+K"].max(), df["Cl"].max())

    fig1.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', line=dict(color='red', dash='dash'), name="Relación 1:1"))
    # Sumar las concentraciones de Ca y Mg
    df["Ca+Mg"] = df["Ca"] + df["Mg"]

    # Sumar las concentraciones de HCO3 y SO4
    df["HCO3+SO4"] = df["HCO3"] + df["SO4"]

    # Crear gráfico interactivo de dispersión de Ca + Mg vs HCO3 + SO4
    fig2 = px.scatter(df, x="HCO3+SO4", y="Ca+Mg", color="Label", 
                    labels={"HCO3+SO4": "HCO3 + SO4 (meq/L)", "Ca+Mg": "Ca + Mg (meq/L)"},
                    title="Relación Ca + Mg vs HCO3 + SO4 en meq/L", 
                    hover_data=["Sample","Ca+Mg", "HCO3+SO4","Label"],
                    color_discrete_sequence=px.colors.qualitative.Set1)  # Barra de color discreta

    # Agregar la línea 1:1
    min_val = min(df["Ca+Mg"].min(), df["HCO3+SO4"].min())
    max_val = max(df["Ca+Mg"].max(), df["HCO3+SO4"].max())

    fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', line=dict(color='red', dash='dash'), name="Relación 1:1"))

    st.markdown("""
        # Análisis de Relaciones Iónicas en Agua
        ## Relación Ca + Mg vs HCO₃ + SO₄ (en meq/L)

        ### Interpretación:

        Indica el tipo de agua en función de su mineralización y su posible interacción con formaciones geológicas.

        - Si **Ca + Mg ≈ HCO₃ + SO₄**, sugiere que la composición iónica se debe principalmente a procesos de disolución de carbonatos (calizas y dolomitas) y sulfatos (yeso o anhidrita).
        - Si **Ca + Mg < HCO₃ + SO₄**, podría haber una fuente adicional de aniones como HCO₃⁻ (posible infiltración de aguas superficiales).
        - Si **Ca + Mg > HCO₃ + SO₄**, puede indicar influencia de procesos como la precipitación de carbonatos o la entrada de otras fuentes catiónicas (intercambio iónico, evaporación).
    """)
    st.plotly_chart(fig)
    st.divider()
    st.markdown("""
        ## Relación Na + K vs Cl (en meq/L)

        ### Interpretación:

        Permite evaluar el origen del agua y procesos como la evaporación y la intrusión salina.

        - Si **Na + K ≈ Cl**, sugiere que la fuente de sodio y potasio es principalmente la disolución de halita (NaCl) o evaporación intensa.
        - Si **Na + K > Cl**, puede deberse a procesos de intercambio iónico, donde el agua ha cedido Ca²⁺ y Mg²⁺ a cambio de Na⁺ en sedimentos arcillosos.
        - Si **Na + K < Cl**, podría indicar una influencia de contaminación antropogénica o procesos de mezcla con agua más mineralizada.
    """)
    st.plotly_chart(fig1)
    st.divider()
    st.markdown("""
        ## Relación Ca + Mg vs HCO₃ (en meq/L)

        ### Interpretación:

        Permite evaluar el equilibrio carbonático del agua y su interacción con materiales geológicos.

        - Si **Ca + Mg ≈ HCO₃**, indica que la mineralización proviene de la disolución de carbonatos (CaCO₃ y MgCO₃).
        - Si **Ca + Mg > HCO₃**, podría haber una fuente adicional de Ca²⁺ y Mg²⁺ (aporte de sulfatos o evaporitas).
        - Si **Ca + Mg < HCO₃**, podría indicar la influencia de CO₂ atmosférico o la disolución de materia orgánica.
    """)
    st.plotly_chart(fig2)
    st.divider()
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 16:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        # Referencias

        Clark, I. (2015). Groundwater Geochemistry and Isotopes (T. & F. Group (ed.)). CRC Press.

        Ghesquière, O., Walter, J., Chesnaux, R., & Rouleau, A. (2015). Scenarios of groundwater chemical evolution in a region of the Canadian Shield based on multivariate statistical analysis. Journal of Hydrology: Regional Studies, 4, 246–266. https://doi.org/10.1016/j.ejrh.2015.06.004

        Shelton, J. L., Engle, M. A., Buccianti, A., & Blondes, M. S. (2018). The isometric log-ratio (ilr)-ion plot: A proposed alternative to the Piper diagram. Journal of Geochemical Exploration, 190(September 2017), 130–141. https://doi.org/10.1016/j.gexplo.2018.03.003

        Singhal, B. B. S., & Gupta, R. . (2010). Applied Hydrogeology of fractured rocks.

        Yang, J., Liu, H., Tang, Z., Peeters, L., & Ye, M. (2022). Visualization of Aqueous Geochemical Data Using Python and WQChartPy. Groundwater, 60(4), 555–564. https://doi.org/10.1111/gwat.13185
    """ )
#----------------------------------------------------------------------------------------------

if st.session_state.page < total_pages:
    # Navegación entre las páginas
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.button("Anterior", on_click=prev_page)
    with col3:
        st.button("Siguiente️", on_click=next_page)
else:
    # Navegación entre las páginas
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.button("Anterior", on_click=prev_page)
    with col3:
        st.button("Regresar", on_click=reset_page)

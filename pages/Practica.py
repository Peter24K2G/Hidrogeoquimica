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
from wqchartpy import hfed


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
total_pages = 8  # Cambia este número según el número de páginas que necesites
if "page" not in st.session_state:
    st.session_state.page = 1

# Funciones para navegar entre páginas
def next_page():
    if st.session_state.page < total_pages:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1


# Mostrar el breadcrumb
breadcrumb = f"Página {st.session_state.page} de {total_pages}"
st.markdown(f"{breadcrumb}")



# Mostrar el contenido según la página actual
#----------------------------------------------------------------------------------------------
if st.session_state.page == 1:
#----------------------------------------------------------------------------------------------
    st.write("Colocar introduccion a los diagramas")

#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 2:
#----------------------------------------------------------------------------------------------
    mgdf = pd.read_excel('Insumos\CA_SUR.xlsx',sheet_name='Muestreo mgL')
    # st.write(mgdf.head())
    st.markdown("""
        ## Diagrama de Gibbs
        El diagrama de Gibbs (1970) relaciona los sólidos disuletos totales como función de la dominanción de $Na^+$ o $Ca^{2+}$ y $Cl^-$ o $HCO_3^-$, ilustrando los procesos de calidad de agua en una cáscara de nuez. A bajas concentraciones disueltas, el agua lluvia sin mucha reacción geoquímica contribuya como iones dominantes son $Na^+$ y $Cl^-$. Cuando entra en contacto con calcita y silicatos de calcio de rápida disolución, el agua aumenta su contenido relativo de $Ca^{2+}$ y $HCO_3^-$. Si hay evaporación concentra la solución, el $Ca^{2+}$ y el $HCO_3^-$ se pierden por precipitación de $CaCO_3$ y la composición del agua cambia hacia arriba en la figura y vuelve a la composición del mar dominada por $Na^-$ y $Cl^-$.
    """)
    

    # Botón para generar el gráfico
    if st.button("Generar diagrama de Gibbs"):
        # Llamar a la función para generar el gráfico
        mgdf['Label'] = ""
        gibbs.plot(mgdf, unit='mg/L', figname='images/Gibbs diagram', figformat='jpg')

        # Mostrar la imagen generada
        st.image("images/Gibbs diagram.jpg", caption="Diagrama de Gibbs", use_container_width=True)    
    
    st.divider()
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 3:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        ## Diagrama de Piper
        El diagrama de Piper fue introducido en 1944 y ha sido desde entonces una herramienta fundamental para el análisis hidrogeoquímico para la clasificación del agua, la determinación del potencial de mezcla y la identificación de las reacciones químicas que controlan determinado sistema.

        Consiste en un diagrama con dos campos triangulares que representan los aniones ($SO_4^{2+}$ y $Cl^-$) y los cationes ($Ca^{2+}$ y $Mg^{2+}$) por separado, y $Na^+$ y $K^+$ y la alcalinidad agrupados. Una vez ubicados los cationes y los aniones en cada triángulo, los puntos se proyectan hacia el rombo y donde se intersecta la proyección de los cationes y los aniones. Este punto permite interpretar el tipo de agua.

        A continuación se presentan las características químicas del agua subterránea en las diferentes zonas del diagrama de Piper:

    """)

    # Cargar las imágenes usando st.image()
    st.image("Fig_Aux/Pipper_expl.JPG", caption="Diagrama de Piper", use_container_width=True)

    st.markdown("""
        y los tipos o facies del agua sobre el diagrama:
    """)

    # Otra imagen
    st.image("Fig_Aux/Pipper_expl_2.JPG", caption="Tipos de Facies de Agua", use_container_width=True)

    st.markdown("""
        o resumidos en una tabla:
    """)

    # Otra imagen
    st.image("Fig_Aux/Facies.JPG", caption="Facies en el Diagrama de Piper", use_container_width=True)

    st.markdown("""
        Recientemente, algunos autores han propuesto la inclusión de técnicas estadísticas para su construcción y mejora, buscando visualizar información adicional como las relaciones $Ca^{2+}$/$Mg^{2+}$ vs $Cl^{-}$/$SO_4^{2}$, útil para identificar condiciones de intercambio catiónico.
    """)

    # Última imagen
    st.image("Fig_Aux/ILR_Piper.JPG", caption="ILR-Piper", use_container_width=True)

    st.markdown("""
        Tomado de: Shelton et al., 2018.
    """)

    mgdf = pd.read_excel('Insumos\CA_SUR.xlsx',sheet_name='Muestreo mgL')
    # Crear un desplegable para seleccionar el gráfico que se desea generar
    option = st.selectbox(
        "Selecciona el gráfico que deseas generar:",
        ["Seleccione un gráfico", "Diagrama de Piper", "Diagrama de contornos de Piper"]
    )

    # Dependiendo de la selección, generamos y mostramos el gráfico correspondiente
    if option == "Diagrama de Piper":
        # Llamar a la función para generar el gráfico de Piper
        mgdf['Label'] = ""
        triangle_piper.plot(mgdf, unit='mg/L', figname='images/triangle Piper diagram', figformat='jpg')

        # Mostrar la imagen generada
        st.image("images/triangle Piper diagram.jpg", caption="Diagrama de Piper", use_container_width=True)

    elif option == "Diagrama de contornos de Piper":
        # Llamar a la función para generar el gráfico de contornos de Piper
        mgdf['Label'] = ""
        contour_piper.plot(mgdf, unit='mg/L', figname='images/contour-filled Piper diagram', figformat='jpg')

        # Mostrar la imagen generada
        st.image("images/contour-filled Piper diagram.jpg", caption="Diagrama de contornos de Piper", use_container_width=True)
      
#----------------------------------------------------------------------------------------------

elif st.session_state.page == 4:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        ## Diagrama de Durov
        Es un digrama tri-linear donde se grafican las concentraciones de los cationes y aniones mayoritarios en meq/L en dos triangulos separados. Las concentrciones en cada triangulo se proyectan sobre el cuadrado central, el cual representa el caracter químico general de la muestra. También es posible graficar otra caracteristica química como los Sólidos Disueltos Totales, el pH, la conductividad, etc.
    """)

    mgdf = pd.read_excel('Insumos\CA_SUR.xlsx',sheet_name='Muestreo mgL')
     # Botón para generar el gráfico
    if st.button("Generar diagrama de Durov"):
        # Llamar a la función para generar el gráfico
        mgdf['Label'] = ""
        durvo.plot(mgdf, unit='mg/L', figname='images/Durvo diagram', figformat='jpg')
        # Mostrar la imagen generada
        st.image("images/Durvo diagram.jpg", caption="Diagrama de Durov", use_container_width=True)

      
#----------------------------------------------------------------------------------------------

elif st.session_state.page == 5:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        # Diagrama de Gaillardet
        Grafica la relación $Ca^{2+}$/$Na^{+}$ vs $HCO_3^{-}$/$Na^{+}$ y $Ca^{2+}$/$Na^{+}$ vs $Mg^{2+}$/$Na^{+}$, identificando su relación con rocas evaporitas, silicatos o carbonatos.
    """)

    mgdf = pd.read_excel('Insumos\CA_SUR.xlsx',sheet_name='Muestreo mgL')
     # Botón para generar el gráfico
    if st.button("Generar diagrama de Gaillardet"):
        # Llamar a la función para generar el gráfico
        mgdf['Label'] = ""
        gaillardet.plot(mgdf, unit='mg/L', figname='images/Gaillardet diagram', figformat='jpg')        # Mostrar la imagen generada

        st.image("images/Gaillardet diagram.jpg", caption="Diagrama de Gaillardet", use_container_width=True)
#----------------------------------------------------------------------------------------------

elif st.session_state.page == 6:
#----------------------------------------------------------------------------------------------
    st.markdown("""
        # Diagrama de Schoeller
        Es in grafico en escala semilogaritmica, en donde la abscisa (escala aritmetica) se organizan los iones y cationes a distancias equidistantes. Los puntos son unidos con líneas rectas. Dentro de las ventajas de estos diagramas es que se pueden graficar de manera simulatánea diferentes muestras lo que permite su comparación.
    """)

    mgdf = pd.read_excel('Insumos\CA_SUR.xlsx',sheet_name='Muestreo mgL')
     # Botón para generar el gráfico
    if st.button("Generar diagrama de Schoeller"):
        # Llamar a la función para generar el gráfico
        mgdf['Label'] = ""
        schoeller.plot(mgdf, unit='mg/L', figname='images/Schoeller diagram', figformat='jpg')        # Mostrar la imagen generada

        st.image("images/Schoeller diagram.jpg", caption="Diagrama de Schoeller", use_container_width=True)
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 7:
#----------------------------------------------------------------------------------------------
    # Cargar los datos
    medf = pd.read_excel('Insumos/CA_SUR.xlsx', sheet_name='Muestreo meqL')

    # Convertir 'Cluster' a tipo texto (string)
    medf["Cluster"] = medf["Cluster"].astype(str)

    # Sumar las concentraciones de Ca y Mg
    medf["Ca+Mg"] = medf["Ca"] + medf["Mg"]

    # Crear gráfico interactivo de dispersión de Ca + Mg vs HCO3 con colores discretos
    fig = px.scatter(medf, x="HCO3", y="Ca+Mg", color="Cluster", 
                    labels={"HCO3": "HCO3 (meq/L)", "Ca+Mg": "Ca + Mg (meq/L)"},
                    title="Relación Ca + Mg vs HCO3 en meq/L", 
                    hover_data=["Sample","Ca+Mg", "HCO3", "Cluster"],
                    color_discrete_sequence=px.colors.qualitative.Set1)  # Barra de color discreta

    # Agregar la línea 1:1
    min_val = min(medf["Ca+Mg"].min(), medf["HCO3"].min())
    max_val = max(medf["Ca+Mg"].max(), medf["HCO3"].max())

    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', line=dict(color='red', dash='dash'), name="Relación 1:1"))

    # Mostrar gráfico interactivo
    st.plotly_chart(fig)

    st.divider()

    # Sumar las concentraciones de Na y K
    medf["Na+K"] = medf["Na"] + medf["K"]

    # Crear gráfico interactivo de dispersión de Na + K vs Cl con colores discretos
    fig1 = px.scatter(medf, x="Cl", y="Na+K", color="Cluster", 
                    labels={"Cl": "Cl (meq/L)", "Na_K_meq": "Na + K (meq/L)"},
                    title="Relación Na + K vs Cl en meq/L", 
                    hover_data=["Sample","Na+K", "K", "Cluster"],
                    color_discrete_sequence=px.colors.qualitative.Set1)  # Barra de color discreta

    # Agregar la línea 1:1
    min_val = min(medf["Na+K"].min(), medf["Cl"].min())
    max_val = max(medf["Na+K"].max(), medf["Cl"].max())

    fig1.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', line=dict(color='red', dash='dash'), name="Relación 1:1"))

    # Mostrar gráfico interactivo
    st.plotly_chart(fig1)

    st.divider()

    # Sumar las concentraciones de Ca y Mg
    medf["Ca+Mg"] = medf["Ca"] + medf["Mg"]

    # Sumar las concentraciones de HCO3 y SO4
    medf["HCO3+SO4"] = medf["HCO3"] + medf["SO4"]

    # Crear gráfico interactivo de dispersión de Ca + Mg vs HCO3 + SO4
    fig2 = px.scatter(medf, x="HCO3+SO4", y="Ca+Mg", color="Cluster", 
                    labels={"HCO3+SO4": "HCO3 + SO4 (meq/L)", "Ca+Mg": "Ca + Mg (meq/L)"},
                    title="Relación Ca + Mg vs HCO3 + SO4 en meq/L", 
                    hover_data=["Sample","Ca+Mg", "HCO3+SO4"],
                    color_discrete_sequence=px.colors.qualitative.Set1)  # Barra de color discreta

    # Agregar la línea 1:1
    min_val = min(medf["Ca+Mg"].min(), medf["HCO3+SO4"].min())
    max_val = max(medf["Ca+Mg"].max(), medf["HCO3+SO4"].max())

    fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', line=dict(color='red', dash='dash'), name="Relación 1:1"))

    # Mostrar gráfico interactivo
    st.plotly_chart(fig2)
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 8:
#----------------------------------------------------------------------------------------------
    st.markdown("""
    # Diagrama HFE-D
    El **diagrama HFE-D** es una herramienta hidrogeoquímica utilizada para clasificar aguas subterráneas según su composición iónica. Representa la relación entre cationes y aniones, ayudando a identificar el tipo de agua y los procesos geológicos que afectan su calidad, como la disolución de carbonatos o evaporitas.
    """ )
    hfed.plot(df, unit='mg/L', figname='HFE-D diagram', figformat='jpg')
#----------------------------------------------------------------------------------------------

# Navegación entre las páginas
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.button("Anterior", on_click=prev_page)
with col3:
    st.button("Siguiente️", on_click=next_page)

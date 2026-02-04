
  
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 



if "df" not in st.session_state:
    st.session_state.df = None

modulo = st.sidebar.selectbox(
    "Seleccione un Módulo",
    ["Home", "Carga de Datos", "EDA"]
)

if modulo == "Home":

    st.title("Análisis Exploratorio de Datos – Bank Marketing")

    st.markdown("""
    **Objetivo:**  
    Analizar los datos de la campaña de marketing bancario para identificar
    patrones y factores asociados a la aceptación del producto.
    """)

    st.subheader("Autor")
    st.write("""
    - **Nombre:** Paulo Daniel Liu Cáceda  
    - **Curso:** Especialización en Python for Analytics  
    - **Año:** 2026
    """)

    st.subheader("Tecnologías")
    st.write("Python, Pandas, NumPy, Streamlit, Matplotlib")

elif modulo == "Carga de Datos":

    st.header("Carga del Data Set")

    uploaded_file = st.file_uploader(
        "Seleccione un archivo CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente")

    elif st.session_state.df is None:
        st.session_state.df = pd.read_csv("datos/BankMarketing.csv", sep=";")
        st.info("Usando dataset por defecto")

    df = st.session_state.df

    st.dataframe(df, use_container_width=True)
    st.write("Dimensión del DataFrame:", df.shape)

    st.dataframe(
        df.isnull().sum().to_frame("Cantidad de nulos"),
        use_container_width=True
    )

elif modulo == "EDA":

    if  st.session_state.df is None:
        st.warning("Primero cargue un dataset")
        st.stop()

    df = st.session_state.df

    def info_general():
        st.header("Información General del Data Set")
        st.dataframe(df.head(), use_container_width=True)
        st.write("Dimensión:", df.shape)
        st.write("Total de valores nulos:", df.isnull().sum().sum())

    def clasificacion_variables():
        st.header("Clasificación de Variables")
        st.dataframe(df.dtypes.to_frame("Tipo de dato"), use_container_width=True)

    def estadisticas_descriptivas():
        
        numeric_cols = df.select_dtypes(include="number").columns

        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

        stats = pd.DataFrame({
            "mean": df[numeric_cols].mean(),
            "median": df[numeric_cols].median(),
            "std": df[numeric_cols].std(),
            "min": df[numeric_cols].min(),
            "max": df[numeric_cols].max(),
            "missing": df[numeric_cols].isnull().sum()
         })

        st.dataframe(stats.round(2), use_container_width=True)

        for col in numeric_cols:
            media = stats.loc[col, "mean"]
            mediana = stats.loc[col, "median"]
            dispersion = stats.loc[col, "std"]

            if media > mediana:
                sesgo = "Sesgada a la derecha"
            elif media < mediana:
                sesgo = "Sesgada a la izquierda"
            else:
                sesgo = "Simétrica"

            st.write(f"{col}: Media={media:.2f}, Mediana={mediana:.2f}, Dispersión={dispersion:.2f} → {sesgo}")

            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot de {col}")
            st.pyplot(fig)



    def valores_faltantes():
        st.header("Análisis de Valores Faltantes")

        nulos = df.isnull().sum()
        porcentaje = (nulos / len(df)) * 100

        resumen = pd.DataFrame({
            "Nulos": nulos,
            "Porcentaje (%)": porcentaje
        }).sort_values("Nulos", ascending=False)

        st.subheader("Conteo de valores faltantes")
        st.dataframe(resumen.round(2), use_container_width=True)

        st.subheader("Visualización simple")
        cols_con_nulos = resumen[resumen["Nulos"] > 0]

        if not cols_con_nulos.empty:
            fig, ax = plt.subplots()
            cols_con_nulos["Nulos"].plot(kind="bar", ax=ax)
            ax.set_ylabel("Cantidad de nulos")
            ax.set_title("Valores faltantes por variable")
            st.pyplot(fig)
        else:
            st.info("No se encontraron valores faltantes en el dataset")

        st.subheader("Discusión breve")
        st.write(
            "Las variables con mayor cantidad de valores faltantes podrían afectar "
            "los análisis posteriores. Dependiendo del porcentaje de nulos, se evaluará "
            "la imputación de valores, el uso de técnicas estadísticas robustas o la "
            "eliminación de variables con alta proporción de datos faltantes."
        )

    def distribucion_variables():
        st.header("5. Distribución de Variables")

        col = st.selectbox(
            "Variable numérica",
            df.select_dtypes(include=np.number).columns
        )

        st.subheader("Histograma")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

        st.subheader("Interpretación visual")
        st.write(
            "El histograma permite observar la forma de la distribución de la variable. "
            "La curva KDE ayuda a identificar si la distribución es simétrica, sesgada "
            "o presenta múltiples modas. Estos patrones son útiles para evaluar supuestos "
            "estadísticos y detectar posibles valores atípicos."
        )

    def variables_categoricas():
        st.header("Análisis de Variables Categóricas")
        col = st.selectbox(
            "Variable categórica",
            df.select_dtypes(exclude=np.number).columns
        )

        conteos = df[col].value_counts()
        proporciones = df[col].value_counts(normalize=True) * 100

        resumen = pd.DataFrame({
            "Frecuencia": conteos,
            "Proporción (%)": proporciones
        })

        st.subheader("Conteos y proporciones")
        st.dataframe(resumen.round(2), use_container_width=True)

        st.subheader("Gráfico de barras")
        fig, ax = plt.subplots()
        sns.barplot(x=conteos.values, y=conteos.index, ax=ax)
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel(col)
        ax.set_title(f"Distribución de {col}")
        st.pyplot(fig)

    def analisis_bivariado():
        st.header("Análisis Bivariado")
        num_cols = df.select_dtypes(include=np.number).columns

        x = st.selectbox("Variable X", num_cols)
        y = st.selectbox("Variable Y", num_cols)

        st.subheader("Gráfico de dispersión")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x], y=df[y], ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Relación entre {x} y {y}")
        st.pyplot(fig)

        st.subheader("Correlación")
        corr = df[[x, y]].corr().iloc[0, 1]
        st.write(f"Coeficiente de correlación de Pearson: **{corr:.2f}**")

        st.subheader("Interpretación visual")
        st.write(
            "El gráfico de dispersión permite evaluar la relación entre ambas variables. "
            "Un patrón ascendente o descendente sugiere correlación positiva o negativa, "
            "mientras que una nube de puntos dispersa indica una relación débil o nula. "
            "El coeficiente de correlación cuantifica la intensidad y dirección de dicha relación."
        )
       

    def analisis_c_vs_c():
        st.header("Análisis Bivariado (C vs. C)")
        cat_cols = df.select_dtypes(exclude=np.number).columns
        c1 = st.selectbox("Categoría 1", cat_cols)
        c2 = st.selectbox("Categoría 2", cat_cols)
        st.dataframe(pd.crosstab(df[c1], df[c2]), use_container_width=True)

    def analisis_parametros():
        st.header("9. Análisis Basado en Parámetros Seleccionados")

        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns

        if len(num_cols) == 0:
            st.warning("El dataset no contiene variables numéricas")
            return

        default_cols = list(num_cols[:2]) if len(num_cols) >= 2 else list(num_cols)

        columnas = st.multiselect(
            "Seleccione variables numéricas",
            num_cols,
            default=default_cols
        )

        if len(columnas) < 1:
            st.warning("Seleccione al menos una variable")
            return

        st.subheader("Resumen estadístico dinámico")
        st.dataframe(
            df[columnas].describe().round(2),
            use_container_width=True
        )

        st.subheader("Visualización dinámica")
        col_graf = st.selectbox("Variable para visualizar", columnas)

        fig, ax = plt.subplots()
        sns.histplot(df[col_graf].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribución de {col_graf}")
        st.pyplot(fig)

        st.subheader("Interpretación")
        st.write(
            "El análisis se adapta a las variables seleccionadas por el usuario, "
            "permitiendo explorar de forma flexible estadísticas descriptivas y "
            "distribuciones. Esta interactividad facilita la identificación de patrones "
            "y comportamientos relevantes en el conjunto de datos."
        )


    def hallazgos_clave():
        st.header("Hallazgos Clave")

    opciones = {
        "1.Información General del Data Set": info_general,
        "2.Clasificación de Variables": clasificacion_variables,
        "3.Estadísticas Descriptivas": estadisticas_descriptivas,
        "4.Análisis de Valores Faltantes": valores_faltantes,
        "5.Distribución de Variables": distribucion_variables,
        "6.Análisis de Variables Categoricas": variables_categoricas,
        "7.Análisis Bivariado": analisis_bivariado,
        "8.Análisis Bivariado (C vs. C)": analisis_c_vs_c,
        "9.Análisis Basado en Parámetros Seleccionados": analisis_parametros,
        "10. Hallazgos Clave": hallazgos_clave
    }

    seccion = st.selectbox("Sección EDA", opciones.keys())
    opciones[seccion]()
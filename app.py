import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación permite cargar un archivo Excel de estudiantes para procesar sus datos y realizar predicciones con el modelo Bagging optimizado.")

# Definir rutas relativas de los artefactos para asegurar compatibilidad en GitHub/local
SCALER_PATH = 'min_max_scaler.joblib'
ONE_HOT_PATH = 'one_hot_columns.joblib'
MODEL_PATH = 'bagging_optimizado.joblib'

# Verificar la existencia de los archivos necesarios
missing_files = [f for f in [SCALER_PATH, ONE_HOT_PATH, MODEL_PATH] if not os.path.exists(f)]
if missing_files:
    st.warning(f"Faltan archivos necesarios para el correcto funcionamiento en el repositorio: {', '.join(missing_files)}")

st.header("1. Carga de Datos")
subir_archivo = st.file_uploader("Sube tu archivo de datos Excel (.xlsx)", type=["xlsx"])

if subir_archivo is not None:
    try:
        # Cargar los datos provistos por el usuario
        df_input = pd.read_excel(subir_archivo)
        st.subheader("Vista previa de los datos cargados")
        st.dataframe(df_input.head())
        
        if st.button("Procesar y Realizar Predicciones"):
            # 1. Copiar y limpiar variables que no se requieren
            df_procesado = df_input.copy()
            
            columnas_a_eliminar = ['ID', 'Año - Semestre', 'Nota_final', 'Aprobo']
            df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
            
            # 2. Cargar lista de columnas One-Hot
            columnas_one_hot = joblib.load(ONE_HOT_PATH)
            
            # Aplicar la codificación One-Hot sobre la variable 'Felder'
            if 'Felder' in df_procesado.columns:
                for col in columnas_one_hot:
                    if col.startswith('Felder_'):
                        categoria = col.replace('Felder_', '')
                        df_procesado[col] = df_procesado['Felder'].apply(lambda x: 1.0 if str(x).strip().lower() == categoria.lower() else 0.0)
                df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
            else:
                st.error("La columna 'Felder' es requerida en el archivo para realizar el procesamiento.")
                st.stop()
            
            # 3. Escalar 'Examen_admisión' utilizando el scaler guardado
            scaler = joblib.load(SCALER_PATH)
            
            if 'Examen_admisión' in df_procesado.columns:
                # El scaler espera forma 2D
                admision_scaled = scaler.transform(df_procesado[['Examen_admisión']])
                df_procesado['Examen_admission_scaled'] = admision_scaled
                df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')
            else:
                st.error("La columna 'Examen_admisión' es requerida en el archivo para realizar el procesamiento.")
                st.stop()
            
            # Alinear las columnas para que coincidan exactamente con la estructura que el modelo espera
            columnas_finales = [col for col in columnas_one_hot if col in df_procesado.columns]
            df_procesado = df_procesado[columnas_finales]
            
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_procesado.head())
            
            # 4. Cargar modelo y predecir
            modelo_bagging = joblib.load(MODEL_PATH)
            predicciones = modelo_bagging.predict(df_procesado)
            
            # Agregar predicciones al DataFrame original para descargarlo
            df_final = df_input.copy()
            df_final['Prediccion_Nota_Final'] = predicciones
            
            st.subheader("Resultados de la Predicción")
            st.dataframe(df_final[['ID', 'Año - Semestre', 'Felder', 'Examen_admisión', 'Prediccion_Nota_Final'] if 'ID' in df_final.columns else df_final.columns])
            
            # Opción para descargar los resultados
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar predicciones como CSV",
                data=csv,
                file_name='predicciones_aprobacion.csv',
                mime='text/csv',
            )
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo de Excel para comenzar.")

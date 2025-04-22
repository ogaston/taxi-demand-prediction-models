# **Predicción de la Demanda de Taxis con Redes Neuronales**  

## **📌 Descripción del Proyecto**  
Este proyecto busca **predecir la demanda de taxis** en una ciudad utilizando datos históricos de viajes, condiciones climáticas y festividades. Se implementó una **red neuronal secuencial** para optimizar la distribución de taxis y reducir los tiempos de espera.  

### **🔍 Contexto**  
- **Problema**: La creciente población urbana dificulta cubrir la demanda de taxis de manera oportuna, lo que genera:  
  - Aumento del tiempo de espera (hasta un 300% en horas pico).  
  - Incremento en tarifas dinámicas (hasta 5x en apps de transporte).  
  - Pérdidas económicas (estimadas en 500,000 USD anuales en NYC).  
- **Solución**: Un modelo de aprendizaje automático que predice la demanda en tiempo real, integrando variables como clima, ubicación y festivos.  

---

## **🚖 Dataset**

Este proyecto utiliza un dataset de viajes de taxis en Nueva York durante el año 2015/2016.

Link al dataset: [yellow_tripdata_2015.csv](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data)

Nota: El dataset debe ser descargado y guardado en la carpeta `Data/`

## **📂 Estructura del Proyecto**  
```  
.  
├── Data/  
│   ├── taxis_dataset_final.csv          # Dataset principal (viajes, clima, festivos)  
│   ├── dataset_clean.py                # Script para limpieza de datos  
│   ├── join_datasets.py                # Combina datos de múltiples fuentes  
│   ├── add_extra_data.py               # Añade variables externas (ej. clima)  
│   ├── update_values.py                # Actualiza valores inconsistentes  
│   └── PlotData.py                     # Genera visualizaciones exploratorias  
├── NeuralNetwork/  
│   ├── TaxiNeuralNetwork.py            # Entrenamiento del modelo  
│   └── TrainedNeuralNetwork.py         # Modelo preentrenado para inferencia  
└── README.md  
```  

---

## **🔧 Instalación y Uso**  

### **1. Requisitos**  
- Python 3.8+  
- Bibliotecas:  
  ```bash  
  pip install pandas numpy matplotlib seaborn tensorflow scikit-learn plotly  
  ```  

### **2. Ejecución**  
- **Preprocesamiento de datos**:  
  ```bash  
  python Data/dataset_clean.py  
  python Data/join_datasets.py  
  ```  
- **Entrenamiento del modelo**:  
  ```bash  
  python NeuralNetwork/TaxiNeuralNetwork.py  
  ```  
- **Predicciones con el modelo entrenado**:  
  ```bash  
  python NeuralNetwork/TrainedNeuralNetwork.py  
  ```  

---

## **📊 Datos y Metodología**  

### **Dataset**  
- **Variables principales**:  
  - **Temporales**: `año`, `mes`, `día`, `hora`.  
  - **Geográficas**: `pickup_longitude`, `pickup_latitude`.  
  - **Externas**: `is_holiday` (festivos), `temperature` (°C), `precipitation` (mm).  
  - **Target**: `demand` (número de viajes).  

### **Preprocesamiento**  
1. **Limpieza**: Eliminación de valores nulos y outliers.  
2. **Integración**: Fusión de datos de clima y festivos.  
3. **Normalización**: Escalado de características numéricas.  

---

## **🤖 Modelo de Red Neuronal**  

### **Arquitectura**  
- **Tipo**: Red neuronal secuencial (Keras).  
- **Capas**:  
  - **Entrada**: 8 neuronas (features de entrada).  
  - **Ocultas**: 3 capas densas (10, 20, 30 neuronas) con activación **ReLU**.  
  - **Salida**: 1 neurona (regresión lineal).  
- **Hiperparámetros**:  
  - **Optimizador**: Adam.  
  - **Función de pérdida**: MSE (Error Cuadrático Medio).  
  - **Épocas**: 50.  
  - **Batch size**: 32.  
---

## **📜 Licencia**  
MIT License.  

--- 

**👥 Autores**:  
- Omar Gastón  
- Oscar Castelblanco  
- Estefanía Bermúdez  

--- 

*Proyecto desarrollado para la clase de Sistemas Inteligentes, Pontificia Universidad Javeriana (2025-1).*
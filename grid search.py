import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load the dataset from the specified file path."""
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    return data

def filter_data(data):
    """Apply filtering conditions on the data."""
    filtered_data = data[(data['Score'] != 0)]
    filtered_data = filtered_data[filtered_data['SEXO'].isin(['M', 'F'])]
    filtered_data = filtered_data[(filtered_data['Edad'] <= 90) & (filtered_data['Edad'] >= 18)]
    filtered_data = filtered_data[(filtered_data['INGRESO_CLIENTE'] <= 500000001) & (filtered_data['INGRESO_CLIENTE'] >= 1000000)]
    filtered_data = filtered_data[(filtered_data['Cant. Cuotas'] <= 24) & (filtered_data['Cant. Cuotas'] >= 1)]
    filtered_data = filtered_data[(filtered_data['Capital actual'] <= 30000000) & (filtered_data['Capital actual'] >= 300000)]
    filtered_data = filtered_data[(filtered_data['Valor Cuota'] <= 10000000) & (filtered_data['Valor Cuota'] >= 50000)]
    filtered_data = filtered_data[filtered_data['Banca'].isin([240, 420, 130, 471, 421, 470])]
    filtered_data = filtered_data[filtered_data['Tipo'].isin([201, 205, 300, 305, 200])]
    return filtered_data

def drop_columns(data, columns_to_drop):
    """Drop unnecessary columns from the data."""
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    return data.drop(columns=columns_to_drop)

def preprocess_data(data, categorical_cols, numerical_cols, target_col):
    """Encode categorical variables, handle missing values, and scale numerical features."""
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    if target_col not in data_encoded.columns:
        logging.error(f"Target column '{target_col}' not found in data columns: {data_encoded.columns}")
        raise KeyError(f"Target column '{target_col}' not found in data columns")

    X = data_encoded.drop(columns=[target_col])
    y = data_encoded[target_col]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, imputer, scaler

def perform_grid_search(X, y):
    """Perform grid search with cross-validation on RandomForest and LogisticRegression models."""
    models = {
        'RandomForest': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000)
    }

    param_grid = {
        'RandomForest': {
            'n_estimators': [100, 200,2000],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'LogisticRegression': {
            'C': [0.01, 0.015, 0.02, 0.05],
            'solver': ['liblinear', 'lbfgs']
        }
    }

    best_models = {}
    for name, model in models.items():
        logging.info(f"Performing grid search for {name}...")
        grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        best_models[name] = grid_search.best_estimator_
        logging.info(f"Best parameters for {name}: {grid_search.best_params_}")
        logging.info(f"Best score for {name}: {grid_search.best_score_}")

    return best_models

def save_model(model, imputer, scaler, model_path, imputer_path, scaler_path):
    """Save the trained model, imputer, and scaler to disk."""
    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    logging.info("Model, imputer, and scaler saved successfully.")

def main():
    # File paths
    train_file_path = '/Users/fabrizioferrari/Desktop/final boss/Training_Set_Columnas_Condensed.csv'
    model_path = '/Users/fabrizioferrari/Desktop/final boss/best_model.pkl'
    imputer_path = '/Users/fabrizioferrari/Desktop/final boss/imputer.pkl'
    scaler_path = '/Users/fabrizioferrari/Desktop/final boss/scaler.pkl'

    # Columns to drop
    columns_to_drop = [
        "Año de Fecha Cierre", "Fecha Cierre", "Fecha Colocacion", "Analista", 
        "APORTA_IVA", "Aportaips", "Aproblinea", "Aprobscoring", "Atraso", 
        "CALIFICACION", "CIUDADLAB", "CIRCUITO_OPE", "CLIENTEFORMAL", "COBROWALTON", "COD_EMPRESA1_LAB", 
        "COD_EMPRESA2_LAB", "Condicionado", "Controlscoring", "CUENTA", 
        "CUOTAS_PEND", "CUOTASPAGADAS", "EMPRESA_PUBLICA1_LAB", 
        "EMPRESA_PUBLICA2_LAB", "EMPRESA1_LAB", "EMPRESA2_LAB", 
        "ESTADO_OPERACION", "Excepcion", "EXCEPCIONANALISTA", 
        "EXCEPCIONESTADO", "EXCEPCIONINSTANCIA", "EXCEPCIONMOTIVO", 
        "EXCEPCIONTIPO", "Faja", "Fecha Venta", "FECHA_CANCELACION", 
        "Franquicia", "HABILITA_PROD1_BNF", "HABILITA_PROD2_BNF", 
        "INTERES_VTA", "INTERES2", "IVA_LEY", "MONTO_ANTERIOR", 
        "MONTODESEMBOLSADO", "OPE_NUEVA", "OPEPARALELA", "Operacion", 
        "OPERACIONIPS", "PATENTE_COMERCIAL", "Rechazocarga", "RUC_EMPRESA1_LAB", 
        "RUC_EMPRESA2_LAB", "Saldo Capital", "Score", "SCORE_BICSA", 
        "SCORE_DATALAB", "SECTOR_ECONOMICO", "SITUACION", "Sucursal", 
        "Sucursaltipo", "Supervisor", "Tipo_Aprobacion", "ULTIMO_ATRASO", 
        "Vendedor", "Pagare", "Mora Final %", "Mora Final", "Atraso30", 
        "Atraso60", "Atraso120", "Atraso150", "Atraso90", "Capital anterior", "Capital Venta"
    ]

    # Categorical and numerical columns
    categorical_cols = [
        "Banca", "CALIFICACION_ANTERIOR", "MARCA", "SEXO", 
        "Tipo", "Departamento", "Medio", "Canal"
    ]
    numerical_cols = [
        "Cant. Cuotas", "Capital actual", "Edad", "INGRESO_CLIENTE", "Valor Cuota"
    ]

    target_col = 'Atraso180'

    # Load and preprocess data
    logging.info("Loading data...")
    train_data = load_data(train_file_path)
    logging.info(f"Columns in the dataset after loading: {train_data.columns.tolist()}")

    logging.info("Filtering data...")
    train_data = filter_data(train_data)

    logging.info("Dropping unnecessary columns...")
    train_data = drop_columns(train_data, columns_to_drop)
    logging.info(f"Columns in the dataset after dropping: {train_data.columns.tolist()}")

    logging.info("Preprocessing data...")
    X_train, y_train, imputer, scaler = preprocess_data(train_data, categorical_cols, numerical_cols, target_col)

    # Perform grid search and train the models
    logging.info("Performing grid search...")
    best_models = perform_grid_search(X_train, y_train)

    # Save the best model, imputer, and scaler
    for model_name, model in best_models.items():
        logging.info(f"Saving the best {model_name} model...")
        save_model(model, imputer, scaler, f'/Users/fabrizioferrari/Desktop/final boss/{model_name}_model.pkl', imputer_path, scaler_path)

    logging.info("Model training completed and saved.")

if __name__ == "__main__":
    main()

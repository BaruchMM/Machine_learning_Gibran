import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class ETL:
    def __init__(self, filepath, add_header=False, remove_columns:list=None):
        self.filepath = filepath
        self.add_header = add_header
        self.remove_columns = remove_columns
        self._load_data()

    def _load_data(self):
        """
        Carga datos desde un archivo CSV con o sin encabezado, considerando la última columna como la variable objetivo.
        """
        if self.add_header:
            # Si el archivo tiene encabezado, lo cargamos normalmente
            data = pd.read_csv(self.filepath)
        else:
            # Si el archivo no tiene encabezado, asignamos nombres genéricos a las columnas
            data = pd.read_csv(self.filepath, header=None)
            num_columns = data.shape[1]
            column_names = [f'col_{i}' for i in range(num_columns)]
            data.columns = column_names
       
        # print(data)
        

        data = self._clean_data(data)
        # print(data.head())

        self.X = data.iloc[:, :-1].values.astype(int)
        self.y = data.iloc[:, -1].values
        self.y = self.y.astype(int)
        # Reindexar clases para que sean consecutivas desde 0
        _, self.y = np.unique(self.y, return_inverse=True)
        # print(data.iloc[:, -1])
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos eliminando filas con valores faltantes o con strings en cualquier columna.
        """
        if self.remove_columns:
            data = data.drop(columns=self.remove_columns)

        # Convertir todas las columnas a tipo numérico (los strings se vuelven NaN)
        data = data.apply(pd.to_numeric, errors='coerce')
        # Eliminar filas con valores faltantes
        cleaned_data = data.dropna()
        return cleaned_data
    
    def _split_data(self, X, y, train_size, validation_size, test_size):
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba según los tamaños especificados
        y con un muestreo estratificado, además de fijar la semilla del generador de números pseudoaleatorios.
        """
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
        )
        validation_ratio = validation_size / (train_size + validation_size)
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_temp, y_temp, test_size=validation_ratio, stratify=y_temp, random_state=42
        )
        
        return X_train, y_train, X_validation, y_validation, X_test, y_test
        
    def get_data(self, train_size, validation_size, test_size):
        """
        Divide en conjuntos de entrenamiento, validación y prueba.
        """
        return self._split_data(self.X, self.y, train_size, validation_size, test_size)
    
    def get_resume(self, etiquetas:dict=None):
        n_clases = len(np.unique(self.y))
        clases = np.unique(self.y)
        n_atrib = self.X.shape[1]
        print(f"#################################################")
        print(f"#           Resumen de datos cargados           #")
        print(f"#################################################\n")
        print(f"Cantidad de clases: {n_clases}")
        print(f"Cantidad de atributos: {n_atrib}")
        for i in clases:
            ratio = np.sum(self.y == i) / len(self.y)*100
            if etiquetas:
                print(f"Proporción de clase '{etiquetas[int(i)]}': {ratio:.2f}%")
            else:
                print(f"Proporción de clase {i}: {ratio:.2f}%")
        print("#################################################")

# Tarea 1
Este poryecto está dividido en módulos para facilitar la comprención y limitar las tareas, entonces quedó dividido en las siguientes clases:
* ETL: clase para estandarizar los datos y hacer limpieza, dejandolos listos para el clasificador.
* CategoricalNB: es el clasificador bayesiano ingenuo.
* ClasificadoresPipeline: flujo general que recibe los conjuntos de datos limpios y separados del ETL instanciado para el conjunto de datos, y hace el flujo de entrenamiento y validación de ambos clasificadores

## Archivos ejecutables
Las clases anteriores ya están conectadas y se puede ejecutar todo el proceso desde los siguientes scripts:
* spam.py: ejectuta todo el proceso para los datos de spam.csv
* cancer.py: ejectuta todo el proceso para los datos de cancer.csv
* aves.py: ejectuta todo el proceso para los datos de aves.csv

## Archivos no ejecutables pero necesarios
Los siguientes archivos son necesarios para el correcto funcionamiento.

Datos:
* spam.csv
* cancer.csv
* aves.csv

Módulos de clases:
* ETL.py
* ClasificadorBayesianoIngenuo.py
* ClasificadoresPipeline.py

****
****
**Nota super importante: en caso de ser necesario, se puede modificar la ruta en cada script ejecutable para enlazarlo al csv en cuestión.**
****
****
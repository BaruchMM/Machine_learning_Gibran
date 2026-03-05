from ETL import ETL
from ClasificadoresPipeline import ClasificadoresPipeline

etl = ETL("Tarea1/cancer.csv", add_header=True, remove_columns=['Código de la muestra'])
test_size = 0.2
validation_size = 0.2
train_size = 1 - test_size - validation_size

etiquetas = {
    0: "Benigno",
    1: "Maligno"
}
etl.get_resume(etiquetas)

X_train, y_train, X_validation, y_validation, X_test, y_test = etl.get_data(train_size, validation_size, test_size)

pipeline = ClasificadoresPipeline(X_train, y_train, X_validation, y_validation, X_test, y_test)

pipeline.ejecutar_pipeline()


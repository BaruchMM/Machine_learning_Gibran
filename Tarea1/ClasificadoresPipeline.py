from ClasificadorBayesianoIngenuo import CategoricalNB
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ClasificadoresPipeline:
    def __init__(self, X_train, y_train, X_validation, y_validation, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.X_test = X_test
        self.y_test = y_test


    # Métodos privados
    def _entrenar_clasificador_bayesiano_ingenuo(self):
        self.clasificador_bayesiano_ingenuo = CategoricalNB()
        self.clasificador_bayesiano_ingenuo.fit(self.X_train, self.y_train)

    def _entrenar_clasificador_gaussian_nb(self):
        self.clasificador_gaussian_nb = GaussianNB()
        self.clasificador_gaussian_nb.fit(self.X_train, self.y_train)

    def _eval_CNB(self, X, y):
        prob_clase = self.clasificador_bayesiano_ingenuo.prob_clase_()
        prob_cond_clase = self.clasificador_bayesiano_ingenuo.prob_cond_clase_(X)
        predicciones = np.argmax(prob_clase + prob_cond_clase, axis=1)
        accuracy = np.mean(predicciones == y)
        matriz = self.get_confusion_matrix(y, predicciones)
        return accuracy, matriz, predicciones

    def _eval_GNB(self, X, y):
        predicciones = self.clasificador_gaussian_nb.predict(X)
        accuracy = np.mean(predicciones == y)
        matriz = self.get_confusion_matrix(y, predicciones)
        return accuracy, matriz, predicciones

    # Método público principal
    def ejecutar_pipeline(self):
        print("Entrenando clasificadores...")
        self._entrenar_clasificador_bayesiano_ingenuo()
        self._entrenar_clasificador_gaussian_nb()

        print("\nEvaluación en entrenamiento:")
        acc_CNB_train, cm_CNB_train, _ = self._eval_CNB(self.X_train, self.y_train)
        acc_GNB_train, cm_GNB_train, _ = self._eval_GNB(self.X_train, self.y_train)
        print(f"CategoricalNB - Exactitud: {acc_CNB_train:.4f}\nMatriz de confusión:\n{cm_CNB_train}")
        print(f"GaussianNB - Exactitud: {acc_GNB_train:.4f}\nMatriz de confusión:\n{cm_GNB_train}")

        print("\nEvaluación en validación:")
        acc_CNB_val, cm_CNB_val, _ = self._eval_CNB(self.X_validation, self.y_validation)
        acc_GNB_val, cm_GNB_val, _ = self._eval_GNB(self.X_validation, self.y_validation)
        print(f"CategoricalNB - Exactitud: {acc_CNB_val:.4f}\nMatriz de confusión:\n{cm_CNB_val}")
        print(f"GaussianNB - Exactitud: {acc_GNB_val:.4f}\nMatriz de confusión:\n{cm_GNB_val}")

        print("\nComparación de desempeño en validación:")
        if acc_CNB_val > acc_GNB_val:
            print("CategoricalNB tuvo mejor desempeño en validación.")
            mejor = 'CNB'
        elif acc_GNB_val > acc_CNB_val:
            print("GaussianNB tuvo mejor desempeño en validación.")
            mejor = 'GNB'
        else:
            print("Ambos clasificadores tuvieron el mismo desempeño en validación.")
            mejor = 'CNB'

        print("\nEvaluación en prueba (mejor clasificador):")
        if mejor == 'CNB':
            acc, cm, _ = self._eval_CNB(self.X_test, self.y_test)
            print(f"CategoricalNB - Exactitud: {acc:.4f}\nMatriz de confusión:\n{cm}")
        else:
            acc, cm, _ = self._eval_GNB(self.X_test, self.y_test)
            print(f"GaussianNB - Exactitud: {acc:.4f}\nMatriz de confusión:\n{cm}")

    def get_confusion_matrix(self, y_true, y_pred):
        classes = np.union1d(y_true, y_pred)
        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            true_index = np.where(classes == true_label)[0][0]
            pred_index = np.where(classes == pred_label)[0][0]
            confusion_matrix[true_index, pred_index] += 1
        return confusion_matrix
    
    def plot_confusion_matrix(self, confusion_matrix, class_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
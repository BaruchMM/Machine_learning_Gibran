import numpy as np

class CategoricalNB:
  "Clasificador bayesiano ingenuo con características categóricas"

  def prob_clase_(self):
    """
    Calcula probabilidad de la clase
    """
    return self.qc

  def prob_cond_clase_(self, X):
    """
    Calcula probabilidad de atributo dada la clase
    """
    v = np.zeros((X.shape[0], self.clases.size))
    a = np.arange(X.shape[1])
    for k in range(X.shape[0]):
      for i in range(self.clases.size):
        v[k, i] = self.qa[i, a, X[k, a]].sum()
    return v

  def fit(self, X, y, alfa = 2):
    """
    Entrena clasificador bayesiano ingenuo
    """
    # Calcula parámetros de distribución a priori (categórica)
    n = X.shape[0]
    self.clases = np.unique(y)
    self.n_clases = self.clases.size
    self.qc = np.zeros(self.clases.size)
    for i,c in enumerate(self.clases):
      # Misma alfa para todas las categorías
      self.qc[i] = (np.sum(y == c) + alfa - 1) / (n + alfa * self.n_clases - self.n_clases)

    # Escala logarítmica para parámetros de a priori
    self.qc[self.qc == 0] = np.nextafter(0, 1)
    self.qc[self.qc == 1] = np.nextafter(1, 0)
    self.qc = np.log(self.qc)

    # Calcula parámetros de verosimilitud (categórica)
    self.n_atrib = X.shape[1]

    # Presupone mismas categorías en los dos atributos (mismos posibles jugadores)
    self.n_cat = int(np.max(X)) + 1
    self.qa = np.zeros((self.n_clases, self.n_atrib, self.n_cat))

    # Estima parámetros para cada clase, atributo y categoría (máximo a posteriori)

    for idx,c in enumerate(self.clases):
      for i in range(self.n_atrib):
        X_ci = X[y == c, i]
        for j in range(self.n_cat):
          # Misma alfa para todas las categorías
          self.qa[idx, i, j] = (np.sum(X_ci == j) + alfa - 1) / (X_ci.shape[0] + alfa * self.n_cat - self.n_cat)

    # Usa escala logarítmica para parámetros de verosimilitud
    self.qa[self.qa == 0] = np.nextafter(0, 1)
    self.qa[self.qa == 1] = np.nextafter(1, 0)
    self.qa = np.log(self.qa)

  def predict(self, X):
    """
    Predice clases dada un conjunto de datos
    """
    aposteriori = self.prob_clase_() + self.prob_cond_clase_(X)
    return self.clases[np.argmax(aposteriori, axis = 1)], np.max(aposteriori, axis = 1)

  def score(self, X, y):
    """
    Calcula exactitud dado datos
    """
    preds, probs = self.predict(X)
    return np.mean(preds == y) * 100
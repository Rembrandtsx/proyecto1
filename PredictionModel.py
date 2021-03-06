from joblib import load


class Model:

    def __init__(self, columns):
        self.model = load("assets/modelo.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result

    def get_r2(self, x, y):
        result = self.model.score(x, y)
        return result

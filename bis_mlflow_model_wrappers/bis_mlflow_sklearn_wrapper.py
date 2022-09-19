# Load mlflow pyfunc
import mlflow.pyfunc
import numpy as np
import pickle

# Train and save an SKLearn model
sklearn_model_path = "model.pkl"

sklearn_artifacts = {"sklearn_model": sklearn_model_path}

# create wrapper
class SKLearnWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass

    def load_context(self, context):
        self.sklearn_model = pickle.load(open(context.artifacts["sklearn_model"], "rb"))

    def predict(self, model, data):
        return self.sklearn_model.predict(data), self.sklearn_model.predict_proba(data)
        # model_predictions = self.sklearn_model.predict_proba(data)
        # return [[np.max(row), np.argmax(row)] for row in model_predictions]

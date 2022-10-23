import bentoml

from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

#dv = model_ref.custom_objects["dictVectorizer"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("hw7_classifier", runners = [model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(user_data):
    prediction = model_runner.predict.run(user_data)
    print(prediction)
    result = prediction[0]

    return result

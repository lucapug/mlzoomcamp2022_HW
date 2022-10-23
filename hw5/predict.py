import pickle

model_file = 'model1.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

dv_file = 'dv.bin'

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

def predict(customer):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    decision = y_pred >= 0.5

    result = {
        'credit_probability': y_pred,
        'card_given': decision
    }

    return result

customer = {"reports": 0, 
            "share": 0.001694, 
            "expenditure": 0.12, 
            "owner": "yes"
            }

result = predict(customer)

print(result)
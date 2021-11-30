
class Baseline:
    name = 'baseline'
    seed = 2021
    train = "input/train.csv"
    test = "input/test.csv"
    svc_params = {
        'kernel': "trial.suggest_categorical('kernel', ['linear','rbf','poly'])",
        'gamma' : "trial.suggest_loguniform('gamma',1e-5,1e5)",
        'C': "trial.suggest_loguniform('C',1e-5,1e5)",
    }
    amp = True
    parallel = None
    deterministic = False

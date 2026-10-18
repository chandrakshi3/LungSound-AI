from src.predict import load_baseline,ModelUnavailable
def test_missing_model(tmp_path):
 try:load_baseline(tmp_path/'missing.joblib')
 except ModelUnavailable:return
 assert False,'missing model must not return fake output'

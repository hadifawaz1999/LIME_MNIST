from MODEL import MY_MODEL
from DATA import load_data_mnnist,transform_labels
import numpy as np
import shap
import matplotlib.pyplot as plt

xtrain,ytrain,xtest,ytest = load_data_mnnist()
my_model=MY_MODEL(xtrain,ytrain,xtest,ytest,show_summary=False,show_details=True,save_model=False)
my_model.load_model('my_model_3_channels.hdf5')
ypred=my_model.predict()
ypred=np.argmax(ypred,axis=1)
print(my_model.get_score())
my_model.get_explanation()
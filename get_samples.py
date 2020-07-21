from DATA import load_data_mnnist
import matplotlib.pyplot as plt

xtrain,ytrain,xtest,ytest=load_data_mnnist()

for i in range(10):
    plt.imshow(xtest[i])
    image_name="images/on_test_set/originals/test_image_"+str(i)+".png"
    plt.savefig(image_name)
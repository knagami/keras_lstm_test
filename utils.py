 
import numpy as np
import matplotlib.pyplot as plt

def showGrapth(history):


    # acc（精度）, val_acc（バリデーションデータに対する精度）のプロット
    plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
    plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
    plt.title('model accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'], label="loss", ls="-", marker="o")
    plt.plot(history.history['val_loss'], label="val_loss", ls="-", marker="x")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

def  create_model(LOSS_FUNCTION , OPTIMIZER , METRICS , NUM_CLASSES):

    Layers = [ tf.keras.layers.Flatten(input_shape= [28 , 28] , name="input_layer"),
            tf.keras.layers.Dense(300 , activation="relu", name="hiddenLayer1"),
            tf.keras.layers.Dense(100 , activation="relu" , name="hiddenLayer2"),
            tf.keras.layers.Dense(NUM_CLASSES , activation="softmax", name="outputLayer") ]

    model_clf = tf.keras.models.Sequential(Layers)
        
    model_clf.summary()
    
    model_clf.compile(loss=LOSS_FUNCTION , optimizer=OPTIMIZER , metrics=METRICS)


    return model_clf


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def save_plot(history ,plot_name , plots_dir):
    unique_filename = get_unique_filename(plot_name)
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plotPath = os.path.join(plots_dir, unique_filename)
    plt.savefig(plotPath)
    plt.show()


def path_tensorboardlogs(tlog_name , logs_dir):
    unique_filename = get_unique_filename(tlog_name)
    path_to_log_file = os.path.join(logs_dir , unique_filename)
    
    return path_to_log_file
    
    






    
    
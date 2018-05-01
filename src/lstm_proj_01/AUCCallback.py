import keras.callbacks
import sklearn.metrics

class AUCLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        auc = []

    def on_batch_end(self, batch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        auc.append(roc)
        print("AUC for this batch is {}".format(roc))


import numpy as np
import sys
from utils import load_data, preprocess_data
from rnn_rbm import RNNRBM
# import matplotlib.pyplot as plt


if __name__ == '__main__':
    dev, mins = sys.argv[1], sys.argv[2]
    mean, std, train_data, test_data = load_data(dev, mins)
    X_train, y_train = \
        preprocess_data(train_data, window_side=5)
    X_test, y_test = \
        preprocess_data(test_data, window_side=5)
    model = RNNRBM(learning_rate=1e-5, visible_dim=1, \
        hidden_dim=300, rnn_hidden_dim=300, num_epochs=10)
    print ("Start training")
    preds, y, mae = model.train(X_train, y_train)
    print ("Finish training")
    # plt.plot(y, 'b-', label='True values')
    # plt.plot(preds, 'r-', label='Predicted values')
    # plt.legend(loc='upper left')
    # plt.title('Prediction {} \nMAE:{} \n'.format(dev.upper(), str(mae)))
    # plt.show()

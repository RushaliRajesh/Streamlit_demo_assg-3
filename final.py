import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse', False)

def has_regularization(model):
    for layer in model.layers:
        if layer.kernel_regularizer is not None:
            return True
    return False

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    X_new = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(X_new)
    Z = Z.reshape(xx.shape)
    cmap = ListedColormap(['#FF0000', '#00FF00'])
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    st.pyplot()

def create_model(num_layers, num_neurons_list, l1, l2, reg_type):
    model = keras.Sequential()
    for i in range(num_layers):
        print(reg_type, l1,l2)
        if reg_type == 'L1':
            print("\nin l1\n")
            model.add(keras.layers.Dense(num_neurons_list[i], activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1)))
        elif reg_type == 'L2':
            model.add(keras.layers.Dense(num_neurons_list[i], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)))
        else:
            model.add(keras.layers.Dense(num_neurons_list[i], activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(x, y, num_layers, num_neurons_list, batch_size, epoch, l1_reg, l2_reg, reg_type):
    model = create_model(num_layers, num_neurons_list, l1_reg, l2_reg, reg_type)
    
    model.fit(x, y, epochs=epoch, batch_size=batch_size, verbose =0)
    # model.summary()
    return model

def plot_weights(model):
    fig, axs = plt.subplots(len(model.layers)-1, 1, figsize=(8, 8))
    for i, layer in enumerate(model.layers[:-1]):
        weights, biases = layer.get_weights()
        axs[i].hist(np.ravel(weights), bins=50)
        axs[i].set_title(f"Layer {i+1}")
        axs[i].set_xlabel("Weight value")
        axs[i].set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)

# Define the plot_weights_in_one() function
def plot_weights_in_one(model):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, layer in enumerate(model.layers[:-1]):
        weights, biases = layer.get_weights()
        ax.hist(np.ravel(weights), bins=50, alpha=0.5, label=f"Layer {i+1}")
    ax.set_title("Distribution of Weight Values")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

def app():
    st.title("Effect of Regularization on MLP Weights and Decision Boundary")
    
    dataset_name = st.sidebar.selectbox("Select a dataset", ["Breast Cancer", "Digits", "Iris", "Make hastie", "Make moons"])

    if dataset_name == "Breast Cancer":
        dataset = datasets.load_breast_cancer()
        x = dataset.data[:, :2]
        y = dataset.target
    elif dataset_name == "Digits":
        dataset = datasets.load_digits()
        x = dataset.data[:, :2]
        y = dataset.target
    elif dataset_name == "Iris":
        dataset = datasets.load_iris()
        x = dataset.data[:, :2]
        y = dataset.target
    elif dataset_name == "Make hastie":
        dataset = datasets.make_hastie_10_2(n_samples=300)
        x = dataset[0][:, :2]
        y = dataset[1]
    elif dataset_name == "Make moons":
        # x, y = generate_spiral_data(1000, noise=1.0)
        dataset = datasets.make_moons(n_samples=300, shuffle=True, noise=0.4, random_state=42)
        x = dataset[0][:, :2]
        y = dataset[1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=5, value=3, step=1)
    num_neurons_per_layer = []
    for i in range(num_layers):
        num_neurons = st.sidebar.slider(f"Number of neurons in layer {i+1}", 1, 50, 10)
        num_neurons_per_layer.append(num_neurons)
    # l1_reg = st.sidebar.slider("L1 regularization", 0.0, 1.0, 0.0)
    # l2_reg = st.sidebar.slider("L2 regularization", 0.0, 1.0, 0.0)

    reg_type = st.sidebar.selectbox("Regularization Type", ["None", "L1", "L2"])
    if reg_type != "None":
        reg_strength = st.sidebar.slider(f"{reg_type} Regularization Strength", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    else:
        reg_strength = 0.0
    l1_reg = 0.0
    l2_reg = 0.0
    if reg_type == "L1":
        l1_reg = reg_strength
    elif reg_type == "L2":
        l2_reg = reg_strength

    epochs = st.sidebar.slider("Number of epochs", 10, 100, 50)
    batch_size = st.sidebar.slider("Batch size", 1, 100, 8)

    model = train_model(x_train, y_train, num_layers, num_neurons_per_layer, batch_size, epochs, l1_reg, l2_reg, reg_type)
    print(model)

    # if has_regularization(model):
    #     print('Regularization has been applied in the model')
    # else:
    #     print('Regularization has not been applied in the model')
    # train_model(x_train, y_train, model, epochs, batch_size)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    st.write("Test Accuracy: {:.2f}%".format(test_accuracy))

    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    st.write("Train Accuracy: {:.2f}%".format(train_accuracy))

    plot_decision_boundary(model, x, y)
    plot_weights(model)
    plot_weights_in_one(model)

if __name__ == '__main__':
    app() 

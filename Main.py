#Imports
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Dataset definition
dataset = keras.datasets.fashion_mnist
((imagens_treino, identificacoes_treino),(imagens_teste, identificacoes_teste)) = dataset.load_data()

# itens titles
nomes_classificacoes = ['camiseta', ' calça', 'pullover',
                         'Vestido', 'Casaco', 'Sandália',
                         'Camisa', 'Tênis', 'Bolsa', 'Bota']

# Exploring data
'''
len(imagens_treino)
print('input treino: ',imagens_treino.shape)
print('input teste: ',imagens_teste.shape)
print(identificacoes_treino.min())
print(identificacoes_treino.max())
total_identificacoes = 10

for imagem in range(10):
    plt.subplot(2, 5, imagem+1)
    plt.imshow(imagens_treino[imagem])
    plt.title(nomes_classificacoes[identificacoes_treino[imagem]])
    plt.show()


# optimizing color range
plt.imshow(imagens_treino[0])
plt.colorbar()
plt.show()
'''

imagens_treino = imagens_treino/float(255)


# Where the children's cry and the mothers don't see
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = tensorflow.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = tensorflow.nn.softmax)
])

# Compiler
modelo.compile(optimizer = 'adam',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])

# Saving and loading the model
modelo.save('modelo.h5')
modelo_salvo = load_model('modelo.h5')
'''
# Graphs to analise results
graph = modelo.fit(imagens_treino, identificacoes_treino, epochs = 5, validation_split = 0.2 )

plt.plot(graph.history['accuracy'])
plt.plot(graph.history['val_accuracy'])
plt.xlabel('acurácia')
plt.ylabel('Épocas')
plt.legend(['treino', 'avaliação'])
plt.show()

plt.plot(graph.history['loss'])
plt.plot(graph.history['val_loss'])
plt.xlabel('Perdas')
plt.ylabel('Épocas')
plt.legend(['Treino', 'Avaliação'])
'''

testes = modelo.predict(imagens_teste)
print('resultado do teste: ', np.argmax(testes[2]))
print('Número da imagem de teste: ', np.argmax(testes[2]))

teste_modelo_salvo = modelo_salvo.predict(imagens_teste)
print('Resultado teste modelo salvo: ', np.argmax(teste_modelo_salvo[1]))

# Evaluating the model
perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print('Perda durante o teste: ', perda_teste)
print('Acurácia do teste: ', acuracia_teste)
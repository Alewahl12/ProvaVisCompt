# Prova de visão computacional

## Descrição do problema

A prova tem como objetivo:
1. **Aplicação de filtros em 6 imagen para cada categoria:** (gatos e cachorros), como filtro gaussiano e equalização de histograma, utilizando OpenCV.
2. **Classificação de imagens por modelo de IA:** treinamento feito com TensorFlow/Keras para dataset CIFAR-10 e usado para classificar imagens em gato ou cachorro.

## Justificativa das técnicas utilizadas

- **CNN (Convolutional Neural Network):** Escolhida por ser comumente utilizada para classificação de imagens, capaz de extrair automaticamente características relevantes;
- **Batch Normalization e Dropout:** Para melhorar a generalização e evitar overfitting;
- **EarlyStopping e ModelCheckpoint:** Para interromper o treinamento ao detectar overfitting e salvar o melhor modelo;
- **OpenCV:** Biblioteca para manipulação e processamento de imagens;
- **Matplotlib:** Para visualização dos resultados obtidos e gráficos.

## Etapas realizadas

1. **app.py**
   - Leitura de imagens da pasta `imagens/`.
   - Aplicação de filtro gaussiano e equalização de histograma.
   - Visualização dos resultados processados.

2. **treinamento.py**
   - Carregamento e normalização do dataset CIFAR-10;
   - Divisão em treino(80%) e teste (20%);
   - Definição e treinamento da CNN;
   - Avaliação do modelo com métricas como o recall e geração de relatório de classificação;
   - Salvamento do modelo treinado e plot de métricas de acuracia e perda.

3. **inferencia.py**
   - Carregamento do modelo treinado.
   - Predição das classes das imagens da pasta `imagens/`.
   - Visualização das imagens com a classificação de cada uma.

## Resultados obtidos

- **Acuracia no CIFAR-10:** 70,94%
- **Relatório de classificação:** Desempenho é decente, com algumas classes sendo mais desafiadoras.
- **Processamento de imagens:** Efeitos de blur e equalização são perceptiveis nas imagens.
- **Inferência:** O modelo conseguiu classificar corretamente pouco mais da metade das imagens, mas para o contexto atual de classificar gatos e cachorros é aceitavel.

## Tempo total gasto para treinar o modelo

- Aproximadamente **91 segundos**.

## Dificuldades encontradas

- Ajuste de hiperparametros para evitar overfitting;
- Classificar imagens externas ao dataset CIFAR-10;
- Tempo de treinamento e limitação de hardware (poderia demorar menos tempo com hardware melhor e se utilizar a GPU).

---

### Estrutura dos arquivos

- [`treinamento.py`](treinamento.py): Treinamento e avaliação da CNN no CIFAR-10.
- [`app.py`](app.py): Processamento e visualização de imagens com filtros.
- [`inferencia.py`](inferencia.py): Inferência do modelo treinado em imagens externas do dataset de treino, presentes na pasta `imagens/`.
- `imagens/`: Pasta com imagens de teste (gatos e cachorros) e imagens que foram utilizadas para aplicar filtros.
- `best_cifar10_tf.h5`: Checkpoint do melhor modelo até então
- `cifar10_cnn_tf_final.h5`: Melhor modelo treinado.
- `requirements.txt`: Arquivo .txt contendo as bibliotecas necessarias para o funcionamento do codigo

---

# 🗑️ Classificação de Resíduos usando Transfer Learning com VGG16

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Latest-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Projeto Final do Curso:** Deep Learning with Keras and TensorFlow (Coursera)  
> **Tecnologia:** Transfer Learning com modelo pré-treinado VGG16  
> **Aplicação:** Classificação automática de resíduos (Orgânico vs. Reciclável)

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Objetivos do Projeto](#objetivos-do-projeto)
- [Características Principais](#características-principais)
- [Dataset](#dataset)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Usar](#como-usar)
- [Resultados](#resultados)
- [Metodologia](#metodologia)
- [Contribuições](#contribuições)
- [Licença](#licença)
- [Referências](#referências)

---

## 🎯 Visão Geral

Este projeto implementa um sistema de classificação automática de resíduos utilizando técnicas avançadas de **Deep Learning** e **Transfer Learning**. O modelo é capaz de distinguir entre resíduos **orgânicos** (O) e **recicláveis** (R) através da análise de imagens, utilizando a arquitetura VGG16 pré-treinada no dataset ImageNet.

### Contexto do Problema

A classificação manual de resíduos é um processo trabalhoso, propenso a erros e que pode levar à contaminação de materiais recicláveis. Este projeto visa automatizar esse processo utilizando visão computacional e aprendizado de máquina, melhorando a eficiência e reduzindo as taxas de contaminação em sistemas de gestão de resíduos.

### Solução Proposta

Utilizando **Transfer Learning** com o modelo VGG16 pré-treinado, desenvolvemos dois modelos:

1. **Extract Features Model**: Modelo que extrai características usando camadas congeladas do VGG16
2. **Fine-Tuned Model**: Modelo refinado com fine-tuning das últimas camadas do VGG16

---

## 🎯 Objetivos do Projeto

Após a conclusão deste projeto, você será capaz de:

- ✅ Aplicar **Transfer Learning** usando o modelo VGG16 para classificação de imagens
- ✅ Preparar e pré-processar dados de imagem para tarefas de machine learning
- ✅ Realizar **fine-tuning** de um modelo pré-treinado para melhorar a acurácia
- ✅ Avaliar o desempenho do modelo usando métricas apropriadas
- ✅ Visualizar predições do modelo em dados de teste

---

## ✨ Características Principais

| Característica | Descrição |
|---------------|-----------|
| **Transfer Learning** | Aproveitamento de conhecimento pré-treinado do VGG16 |
| **Data Augmentation** | Aumento de dados para melhor generalização |
| **Dois Modelos** | Comparação entre extract features e fine-tuning |
| **Otimização GPU** | Configurado para aceleração em Intel Arc GPU |
| **Visualizações** | Gráficos de perda e acurácia durante o treinamento |
| **Model Checkpointing** | Salvamento automático dos melhores modelos |
| **Early Stopping** | Prevenção de overfitting durante o treinamento |
| **Learning Rate Decay** | Decaimento exponencial da taxa de aprendizado |

---

## 📊 Dataset

### Fonte

O dataset utilizado é o [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) disponível no Kaggle.

### Estrutura

```
o-vs-r-split/
├── train/
│   ├── O/          # 500 imagens de resíduos orgânicos
│   └── R/          # 500 imagens de resíduos recicláveis
└── test/
    ├── O/          # 100 imagens de resíduos orgânicos
    └── R/          # 100 imagens de resíduos recicláveis
```

### Estatísticas

| Métrica | Valor |
|---------|-------|
| **Total de imagens** | 1.200 |
| **Treinamento** | 1.000 imagens (800 treino + 200 validação) |
| **Teste** | 200 imagens |
| **Classes** | 2 (Orgânico 'O' e Reciclável 'R') |
| **Dimensões** | 150x150 pixels |
| **Formato** | JPG |

### Divisão dos Dados

- **Treino**: 80% (800 imagens)
- **Validação**: 20% do conjunto de treino (200 imagens)
- **Teste**: 200 imagens (100 por classe)

---

## 🏗️ Arquitetura do Modelo

### Base Model: VGG16

O modelo utiliza a arquitetura VGG16 pré-treinada no ImageNet como base:

```
VGG16 Base (Congelado)
├── Blocos Convolucionais (congelados)
│   ├── Conv2D + ReLU
│   ├── MaxPooling2D
│   └── ... (13 camadas convolucionais)
└── Camadas Densas (treináveis)
    ├── Flatten
    ├── Dense(512) + ReLU + Dropout(0.5)
    ├── Dense(512) + ReLU + Dropout(0.5)
    └── Dense(1) + Sigmoid (saída binária)
```

### Parâmetros do Modelo

| Parâmetro | Valor |
|-----------|-------|
| **Total de parâmetros** | 19,172,673 |
| **Parâmetros treináveis** | 4,457,985 (Extract Features Model) |
| **Parâmetros não-treináveis** | 14,714,688 (camadas congeladas do VGG16) |
| **Tamanho do modelo** | ~73.14 MB |

### Fine-Tuning

No modelo fine-tuned, as últimas camadas convolucionais do VGG16 são descongeladas e treinadas com uma taxa de aprendizado menor.

---

## 🛠️ Tecnologias Utilizadas

### Bibliotecas Principais

- **TensorFlow** 2.17.0 - Framework de deep learning
- **Keras** - API de alto nível para TensorFlow
- **NumPy** 1.26.4 - Operações matemáticas e arrays
- **scikit-learn** 1.5.1 - Métricas de avaliação
- **Matplotlib** 3.9.2 - Visualização de dados e gráficos

### Extensões e Otimizações

- **Intel Extension for TensorFlow** - Aceleração em Intel Arc GPU
- **ImageDataGenerator** - Geração e aumento de dados em tempo real

### Hardware Otimizado

- Intel Arc GPU (opcional, mas recomendado para melhor performance)

---

## 💻 Instalação

### Pré-requisitos

- Python 3.12 ou superior
- pip (gerenciador de pacotes Python)
- Git (para clonar o repositório)

### Passo a Passo

**1. Clone o repositório**

```bash
git clone https://github.com/seu-usuario/waste-classification-vgg16.git
cd waste-classification-vgg16
```

**2. Crie um ambiente virtual (recomendado)**

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

**3. Instale as dependências**

```bash
pip install tensorflow==2.17.0
pip install numpy==1.26.4
pip install scikit-learn==1.5.1
pip install matplotlib==3.9.2
pip install intel-extension-for-tensorflow[xpu]  # Opcional, para Intel Arc GPU
```

**4. Baixe o dataset**

O dataset será baixado automaticamente ao executar o notebook. Alternativamente, você pode baixá-lo manualmente através do código fornecido no notebook.

---

## 📁 Estrutura do Projeto

```
waste-classification-vgg16/
│
├── README.md                                    # Este arquivo
├── Final Proj-Classify Waste Products Using TL- FT-v1.ipynb  # Notebook principal
│
├── o-vs-r-split/                               # Dataset
│   ├── train/
│   │   ├── O/                                  # Imagens orgânicas (treino)
│   │   └── R/                                  # Imagens recicláveis (treino)
│   └── test/
│       ├── O/                                  # Imagens orgânicas (teste)
│       └── R/                                  # Imagens recicláveis (teste)
│
├── O_R_tlearn_vgg16.keras                      # Modelo Extract Features salvo
├── O_R_tlearn_fine_tune_vgg16.keras           # Modelo Fine-Tuned salvo
│
└── venv/                                       # Ambiente virtual (não versionado)
```

---

## 🚀 Como Usar

### Executando o Notebook Completo

**1. Abra o notebook Jupyter**

```bash
jupyter notebook "Final Proj-Classify Waste Products Using TL- FT-v1.ipynb"
```

Ou use VS Code / JupyterLab para abrir o arquivo `.ipynb`

**2. Execute as células sequencialmente**

O notebook está organizado em tarefas numeradas:

- **Task 1**: Verificar versão do TensorFlow
- **Task 2**: Criar gerador de dados de teste
- **Task 3**: Verificar tamanho do gerador de treino
- **Task 4**: Visualizar resumo do modelo
- **Task 5**: Compilar o modelo
- **Task 6**: Plotar curvas de acurácia (Extract Features)
- **Task 7**: Plotar curvas de perda (Fine-Tuned)
- **Task 8**: Plotar curvas de acurácia (Fine-Tuned)
- **Task 9**: Visualizar predições (Extract Features)
- **Task 10**: Visualizar predições (Fine-Tuned)

### Carregando Modelos Pré-treinados

```python
import tensorflow as tf

# Carregar modelo Extract Features
extract_feat_model = tf.keras.models.load_model('O_R_tlearn_vgg16.keras')

# Carregar modelo Fine-Tuned
fine_tune_model = tf.keras.models.load_model('O_R_tlearn_fine_tune_vgg16.keras')
```

### Fazendo Predições

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar e pré-processar imagem
img_path = 'path/to/waste/image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Fazer predição
prediction = fine_tune_model.predict(img_array)
class_label = 'O' if prediction[0][0] < 0.5 else 'R'
print(f"Classificação: {class_label}")
```

---

## 📈 Resultados

### Métricas de Desempenho

Os modelos foram avaliados no conjunto de teste com 200 imagens (100 por classe).

#### Extract Features Model

| Métrica | Valor |
|---------|-------|
| **Acurácia** | ~84-85% |
| **Validação** | Acurácia final de ~84.9% após 10 épocas |
| **Loss de validação** | ~0.36 |

#### Fine-Tuned Model

| Métrica | Valor |
|---------|-------|
| **Acurácia** | Melhorada em relação ao Extract Features Model |
| **Validação** | Acurácia final superior após fine-tuning |
| **Loss de validação** | Reduzida comparado ao modelo base |

### Curvas de Treinamento

O notebook inclui visualizações de:

- **Curvas de Loss**: Treino vs. Validação
- **Curvas de Acurácia**: Treino vs. Validação
- **Comparação entre modelos**: Extract Features vs. Fine-Tuned

### Visualizações de Predições

O projeto inclui visualizações de predições em imagens de teste, mostrando:

- Imagem original
- Classe verdadeira
- Classe predita
- Probabilidade de confiança

---

## 🔬 Metodologia

### 1. Preparação dos Dados

- **Normalização**: Redimensionamento para 150x150 pixels
- **Rescaling**: Normalização de valores de pixel (0-255 → 0-1)
- **Data Augmentation** (treino):
  - Rotação horizontal (horizontal_flip)
  - Deslocamento de largura (width_shift_range=0.1)
  - Deslocamento de altura (height_shift_range=0.1)

### 2. Arquitetura do Modelo

#### Extract Features Model

1. Carregar VGG16 pré-treinado (pesos ImageNet)
2. Congelar todas as camadas convolucionais
3. Adicionar camadas densas personalizadas:
   - Flatten
   - Dense(512) + ReLU + Dropout(0.5)
   - Dense(512) + ReLU + Dropout(0.5)
   - Dense(1) + Sigmoid

#### Fine-Tuned Model

1. Descongelar últimas camadas convolucionais do VGG16
2. Treinar com taxa de aprendizado reduzida
3. Manter camadas densas treináveis

### 3. Treinamento

| Parâmetro | Valor |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 1e-4 inicial com decaimento exponencial |
| **Loss Function** | Binary Crossentropy |
| **Batch Size** | 32 |
| **Epochs** | 10 (com early stopping) |
| **Callbacks** | Early Stopping, Model Checkpoint, Learning Rate Scheduler |

**Callbacks utilizados:**

- Early Stopping (patience=4, monitor='val_loss')
- Model Checkpoint (salvar melhor modelo)
- Learning Rate Scheduler (decaimento exponencial)

### 4. Avaliação

- **Métricas**: Acurácia, Precision, Recall, F1-Score
- **Visualização**: Matriz de confusão, relatórios de classificação
- **Teste**: 200 imagens não vistas durante o treinamento

---

## 🤝 Contribuições

Contribuições são bem-vindas! Se você deseja contribuir para este projeto:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Áreas para Melhorias Futuras

- [ ] Adicionar mais classes de resíduos (papel, plástico, vidro, etc.)
- [ ] Implementar API REST para predições em tempo real
- [ ] Criar interface web para upload de imagens
- [ ] Adicionar suporte para vídeo em tempo real
- [ ] Otimizar modelo para dispositivos móveis (TensorFlow Lite)
- [ ] Implementar ensemble de modelos
- [ ] Adicionar explicações de predições (XAI)

---

## 📄 Licença

Este projeto foi desenvolvido como parte do curso **Deep Learning with Keras and TensorFlow** da Coursera. O código é fornecido para fins educacionais.

---

## 📚 Referências

### Artigos e Documentação

- [VGG16 Paper](https://arxiv.org/abs/1409.1556) - Very Deep Convolutional Networks for Large-Scale Image Recognition
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning) - TensorFlow Official Documentation
- [Keras Applications](https://keras.io/api/applications/vgg/#vgg16-function) - VGG16 Pre-trained Model

### Datasets

- [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) - Kaggle Dataset

### Cursos e Tutoriais

- [Deep Learning with Keras and TensorFlow](https://www.coursera.org/) - Coursera Course
- [Skills Network](https://skills.network/) - IBM Skills Network

### Bibliotecas

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 👤 Autor

**Seu Nome**

- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [Seu Perfil](https://linkedin.com/in/seu-perfil)
- Email: seu.email@example.com

---

## 🙏 Agradecimentos

- **Coursera** e **IBM Skills Network** pelo excelente curso
- **Kaggle** pela disponibilização do dataset
- Comunidade open-source pelas ferramentas e bibliotecas utilizadas

---

## 📧 Contato

Para dúvidas, sugestões ou colaborações, sinta-se à vontade para abrir uma issue ou entrar em contato.

---

<div align="center">

**Se este projeto foi útil para você, considere dar uma estrela! ⭐**

Feito com ❤️ usando TensorFlow e Keras

</div>

# 🌱 Identificação de Pragas e Doenças em Soja com Deep Learning

> Um sistema de Visão Computacional baseado em Transformers para auxiliar na agricultura de precisão.

## 📖 Sobre o Projeto
Este projeto visa automatizar a detecção de danos foliares na cultura da soja. Utilizando um dataset real capturado em campo (com variações de luz, sombra e vento), treinei um modelo de Inteligência Artificial capaz de classificar folhas em três categorias:
1. **Saudável**
2. **Caterpillar** (Danos por lagarta)
3. **Diabrotica Speciosa** (Danos por vaquinha-verde-amarela)

O objetivo é fornecer uma ferramenta para diagnóstico rápido, permitindo controle de pragas mais eficiente e redução do uso indiscriminado de defensivos.

## 🛠️ Tecnologias
* **Modelo**: Vision Transformer (ViT-base-patch16-224) via *Transfer Learning*.
* **Frameworks**: PyTorch, Hugging Face Transformers, Datasets.
* **Acurácia**: XX% (Preencha após rodar o treino).

## 📊 O Dataset
O conjunto de dados consiste em **6.410 imagens** de folhas de soja (500x500px), capturadas via smartphones e drones em ambiente não controlado.
* **Desafio Técnico**: O dataset possui forte desbalanceamento (muitas imagens de lagarta, poucas saudáveis), tratado no código via divisão estratificada.

## 🚀 Como Executar
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`
3. Organize o dataset na pasta `dataset_soja/`.
4. Execute o treino:
   ```bash
   python train_soja.py
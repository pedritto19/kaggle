import os
import pandas as pd
import autogluon.multimodal as agmm
import numpy as np
import cv2

def main():
    # Carregar os dados
    train = pd.read_csv('input/cidaut-ai-fake-scene-classification-2024/train.csv')
    test = pd.read_csv('input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')
    sub = pd.read_csv('input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')

    # Definir os caminhos das imagens
    train_path = 'input/cidaut-ai-fake-scene-classification-2024/Train/'
    test_path = 'input/cidaut-ai-fake-scene-classification-2024/Test/'

    # Codificação binária dos rótulos
    cls_to_idx = {'editada': 0, 'real': 1}
    train['label'] = train['label'].map(cls_to_idx)

    # Atualizar os caminhos das imagens
    train['image'] = train_path + train['image']
    test['image'] = test_path + test['image']

    # Função para calcular a entropia do ruído
    def noise_entropy(im_file):
        image = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        noise = cv2.subtract(image, blurred_image)
        hist, _ = np.histogram(noise.ravel(), bins=256, range=(0, 256))
        hist_prob = hist / hist.sum()
        entropy = -np.sum(hist_prob * np.log2(hist_prob + 1e-7))
        return entropy

    # Função para calcular entropia de todas as imagens
    def get_all_noise_entropy(df):
        return [noise_entropy(path) for path in df['image'].values]

    # Adicionar coluna de entropia
    train['noise_entropy'] = get_all_noise_entropy(train)
    test['noise_entropy'] = get_all_noise_entropy(test)

    # Calcular a média das entropias do conjunto de treinamento
    mean_entropy = train['noise_entropy'].mean()

    # Transformar entropia em valor binário (0 ou 1)
    train['noise_entropy'] = train['noise_entropy'].apply(lambda x: 1 if x > mean_entropy else 0)
    test['noise_entropy'] = test['noise_entropy'].apply(lambda x: 1 if x > mean_entropy else 0)

    # Treinar o modelo usando apenas a entropia
    predictor = agmm.MultiModalPredictor(label='label', eval_metric='auc')
    predictor.fit(train[['noise_entropy', 'label']])

    # Fazer previsões
    preds = predictor.predict(test[['noise_entropy']], as_pandas=False)

    sub['label'] = preds

    sub.to_csv('submissionteste.csv', index=False)

if __name__ == '__main__':
    main()
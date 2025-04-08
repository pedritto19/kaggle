import os
import pandas as pd
import autogluon.multimodal as agmm
from PIL import Image, ImageEnhance, ImageOps, ImageStat
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

    # Função para calcular o brilho
    def brightness(im_file):
        im = Image.open(im_file).convert('L')
        stat = ImageStat.Stat(im)
        return stat.mean[0]

    # Função para calcular o contraste
    def contrast(im_file):
        im = Image.open(im_file)
        enhancer = ImageEnhance.Contrast(im)
        return np.std(np.array(enhancer.enhance(1.0)))

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

    # Funções para calcular brilho, contraste e entropia de todas as imagens
    def get_all_brightness(df):
        return [brightness(path) for path in df['image'].values]

    def get_all_contrast(df):
        return [contrast(path) for path in df['image'].values]

    def get_all_noise_entropy(df):
        return [noise_entropy(path) for path in df['image'].values]

    # Adicionar colunas de brilho, contraste e entropia
    train['brightness'] = get_all_brightness(train)
    train['contrast'] = get_all_contrast(train)
    train['noise_entropy'] = get_all_noise_entropy(train)
    test['brightness'] = get_all_brightness(test)
    test['contrast'] = get_all_contrast(test)
    test['noise_entropy'] = get_all_noise_entropy(test)

    # Treinar o modelo
    predictor = agmm.MultiModalPredictor(label='label', eval_metric='auc')
    predictor.fit(train)

    # Fazer previsões
    preds = predictor.predict(test.drop(columns='label'), as_pandas=False)

    sub['label'] = preds

    sub.to_csv('submissionrate2.csv', index=False)

if __name__ == '__main__':
    main()


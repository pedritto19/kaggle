import pandas as pd
import autogluon.multimodal as agmm
from PIL import Image, ImageStat

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

    # Função para calcular o brilho de todas as imagens
    def get_all_brightness(df):
        return [brightness(path) for path in df['image'].values]

    # Adicionar a coluna de brilho
    train['brightness'] = get_all_brightness(train)
    test['brightness'] = get_all_brightness(test)

    # Treinar o modelo
    predictor = agmm.MultiModalPredictor(label='label', eval_metric='auc')
    predictor.fit(train)

    # Fazer previsões
    preds = predictor.predict(test.drop(columns='label'), as_pandas=False)

    sub['label'] = preds

    sub.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
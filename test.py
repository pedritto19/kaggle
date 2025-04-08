import os
import pandas as pd
import autogluon.multimodal as agmm

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

    # Configurar o MultiModalPredictor
    predictor = agmm.MultiModalPredictor(label='label', eval_metric='auc')
    
    # Treinar o modelo diretamente com imagens
    predictor.fit(train_data=train, time_limit=3600)

    # Fazer previsões
    preds = predictor.predict(test, as_pandas=False)

    sub['label'] = preds

    sub.to_csv('submissionteste.csv', index=False)

if __name__ == '__main__':
    main()
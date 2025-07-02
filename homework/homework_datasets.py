import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from homework_model_modification import LogisticRegressionManual, accuracy
from utils import ClassificationDataset, make_classification_data


class CSVDataset(Dataset):
    def __init__(self, file_path, target_column=None, 
                 numeric_features=None, categorical_features=None, 
                 binary_features=None, scaling='standard', 
                 test_size=0.2, random_state=42, mode='train'):

#        - file_path: путь к CSV файлу
#        - target_column: имя целевой переменной (None если нет)
#        - numeric_features: список числовых признаков
#        - categorical_features: список категориальных признаков
#        - binary_features: список бинарных признаков
#       - scaling: метод масштабирования ('standard', 'minmax' или None)
#        - test_size: размер тестовой выборки (если нужен train/test split)
#        - random_state: seed для воспроизводимости
#        - mode: режим работы ('train', 'test' или 'full')

        # Загрузка данных
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        self.mode = mode
        
        # Определение признаков
        self.numeric_features = numeric_features if numeric_features else []
        self.categorical_features = categorical_features if categorical_features else []
        self.binary_features = binary_features if binary_features else []
        
        # Проверка, что все указанные признаки существуют
        self._validate_features()
        
        # Предобработка данных
        self._preprocess_data(scaling, test_size, random_state)
        
    def _validate_features(self):
        #Проверка, что все указанные признаки существуют в данных
        all_features = self.numeric_features + self.categorical_features + self.binary_features
        if self.target_column:
            all_features.append(self.target_column)
            
        for feature in all_features:
            if feature not in self.data.columns:
                raise ValueError(f"Признак '{feature}' не найден в данных")

    def _preprocess_data(self, scaling, test_size, random_state):
        # Разделение на train/test если нужно
        if test_size > 0 and self.mode in ['train', 'test']:
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                self.data, test_size=test_size, random_state=random_state
            )
            self.data = train_data if self.mode == 'train' else test_data
        
        # Сохраняем target если есть
        self.target = None
        if self.target_column:
            self.target = self.data[self.target_column].values
            self.data = self.data.drop(columns=[self.target_column])
        
        # Создаем трансформеры для разных типов признаков
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler() if scaling == 'standard' else MinMaxScaler())
        ]) if scaling else Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Комбинируем трансформеры
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('binary', binary_transformer, self.binary_features)
            ])
        
        # Применяем трансформации
        self.features = self.preprocessor.fit_transform(self.data)
        
        # Сохраняем имена фичей после OneHot кодирования
        self._get_feature_names()
        
    def _get_feature_names(self):
        #Получаем имена всех фичей после трансформации
        self.feature_names = []
        
        # Числовые признаки
        self.feature_names.extend(self.numeric_features)
        
        # Категориальные признаки (после OneHot)
        if self.categorical_features:
            ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_names = ohe.get_feature_names_out(self.categorical_features)
            self.feature_names.extend(cat_names)
        
        # Бинарные признаки
        self.feature_names.extend(self.binary_features)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        
        if self.target is not None:
            target = torch.tensor(self.target[idx], dtype=torch.float32 if self.target.dtype.kind in 'f' else torch.long)
            return features, target
        return features
    
    def get_feature_names(self):
        #Возвращает имена всех фичей после трансформации
        return self.feature_names
    
    def describe(self):
        #Выводит информацию о датасете
        print(f"Размер датасета: {len(self)}")
        print(f"Количество признаков: {len(self.feature_names)}")
        if self.target is not None:
            print(f"Целевая переменная: {self.target_column}")
            print(f"Классы: {np.unique(self.target)}")
        print("\nПримеры признаков:")
        print(pd.DataFrame(self.features[:5], columns=self.feature_names))

if __name__ == '__main__':
    file_path = 'synthetic_employee_burnout.csv'
    target_col = 'Burnout'
    numeric_cols = ['Age', 'Experience', 'WorkHoursPerWeek']
    categorical_cols = ['Gender', 'JobRole']

    train_dataset = CSVDataset(
    file_path=file_path,
    target_column=target_col,
    numeric_features=numeric_cols,
    categorical_features=categorical_cols,
    mode='train'
)
    
    input_dim = len(train_dataset.get_feature_names())
    n_classes = 1  # Бинарная классификация (Burn Rate - вероятность)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = LogisticRegressionManual(in_features=2, n_classes=n_classes)
    lr = 0.1
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_targets = []
        
        X, y = make_classification_data(n=200)
        dataset = ClassificationDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            
            # Вычисляем loss (cross-entropy)
            if model.n_classes > 2:
                loss = -torch.sum(torch.log(y_pred + 1e-8) * batch_y) / batch_X.shape[0]
            else:
                loss = -(batch_y * torch.log(y_pred + 1e-8) + (1 - batch_y) * torch.log(1 - y_pred + 1e-8)).mean()
            
            acc = accuracy(y_pred, batch_y)
            
            total_loss += loss.item()
            total_acc += acc
            
            # Сохраняем предсказания и метки для метрик
            all_preds.append(y_pred.detach())
            all_targets.append(batch_y.detach())
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
import logging, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer

# Configure the logger
logging.basicConfig(filename='model_retraining.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger instance
logger = logging.getLogger("model_retraining")
df = pd.read_excel("Superstore.xlsx",usecols=['Profit','Postal Code','Discount','Quantity','Order Date','Ship Date','Sales'])

for idx in list(df[df['Sales'] > 10500].index):
    df.drop(idx,axis=0,inplace=True)

df['Order Date'] = pd.to_datetime(df['Order Date'],errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'],errors='coerce')
df['Order Day'] = df['Order Date'].dt.day
df['Order Month'] = df['Order Date'].dt.month
df['Order Weekday'] = df['Order Date'].dt.weekday
df['Ship Day'] = df['Ship Date'].dt.day
df['Discount Percentage'] = df['Discount'] / df['Sales'] * 100
df['Operating Expenses'] = df['Sales'] - df['Profit']
df.drop(['Order Date', 'Ship Date'],axis=1,inplace=True)

transformer = ColumnTransformer(transformers=[
    ('log_transform',FunctionTransformer(np.log1p),['Quantity','Profit']),
    ('sqrt_transform',FunctionTransformer(np.sqrt),['Discount']),
    ('power_transform',PowerTransformer(),['Discount Percentage','Operating Expenses'])
],remainder='passthrough')

X = df.drop('Sales',axis=1)
y = df['Sales']
features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True,random_state=75)

X_train = transformer.fit_transform(X_train)
X_train = pd.DataFrame(X_train,columns=features)
X_test = transformer.transform(X_test)
X_test = pd.DataFrame(X_test,columns=features)

columns_with_missing_values = X_train.columns[X_train.isnull().any()].tolist()

for col in columns_with_missing_values:
    imputer = SimpleImputer(strategy='median')
    X_train[col] = imputer.fit_transform(X_train[[col]])

columns_with_missing_values = X_test.columns[X_test.isnull().any()].tolist()

for col in columns_with_missing_values:
    imputer = SimpleImputer(strategy='median')
    X_test[col] = imputer.fit_transform(X_test[[col]])

winsorizer = Winsorizer(capping_method='iqr',fold=1.5,tail='both')
X_train['Discount Percentage'] = winsorizer.fit_transform(X_train[['Discount Percentage']])
X_test['Discount Percentage'] = winsorizer.transform(X_test[['Discount Percentage']])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_model(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    logger.info("Evaluation metrics - R2 Score: %.2f, Mean Absolute Error: %.2f, Mean Squared Error: %.2f, Mean Absolute Percentage Error: %.2f, Root Mean Squared Error: %.2f", r2, mae, mse, mape, rmse)
    return model, r2

model, baseline_r2 = train_and_evaluate_model(ExtraTreesRegressor())

param_grid = {
    'n_estimators': [200,500,800,1000],
    'criterion': ['squared_error','absolute_error','friedman_mse','poisson'],
    'max_features': ['auto','sqrt','log2'],
    'bootstrap': [True,False],
    'oob_score': [True,False],
    'max_samples': [0.5,0.75,0.9,1]
}

grid_et = RandomizedSearchCV(estimator=ExtraTreesRegressor(),param_distributions=param_grid,cv=5,verbose=2)
optimized_model, optimized_r2 = train_and_evaluate_model(grid_et)

if baseline_r2 < optimized_r2:
    model = optimized_model

avg_cv_scores = cross_val_score(model,X_test,y_test,scoring='r2',cv=5,verbose=2)
mean_score = round(np.mean(avg_cv_scores),2) * 100
logger.info(f"Mean Cross Validation Performance of Extra Trees Regressor: {mean_score}%")

pipeline = Pipeline(steps=[
    ('transformer',transformer),
    ('scaler',scaler),
    ('model',model)
])

logging.shutdown()
joblib.dump(pipeline,'pipeline.pkl')
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, RocCurveDisplay,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_excel('resources/Telco_customer_churn.xlsx')

# Preprocessing Stage
data = data.drop(['Churn Score', "Churn Label", 'CLTV', 'Churn Reason', 'City', 'Lat Long', 'Latitude', 'Longitude', 'State', 'Country', 'Count', 'CustomerID'], axis=1)

premium_services = ['Online Security', 'Online Backup', 'Device Protection', 
                   'Tech Support', 'Streaming TV', 'Streaming Movies']




data["Monthly Tenure Ratio"] = np.where(
    data["Tenure Months"] == 0,
    data['Tenure Months'],
    
    data["Monthly Charges"] / data["Tenure Months"]
)

drop_col = data.pop("Monthly Tenure Ratio")
data.insert(1, "Monthly Tenure Ratio", drop_col)


categorical_cols = [col for col in data.columns 
    if data[col].dtype == 'object' and 1 < len(data[col].unique()) <= 4]
coder = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = coder.fit_transform(data[categorical_cols])
column_names = coder.get_feature_names_out(categorical_cols)
encoded_data = pd.DataFrame(
    encoded_array, 
    columns=column_names, 
    index=data.index
)
data = data.drop(columns=categorical_cols, axis=1)
data = pd.concat([data, encoded_data], axis=1)

temp = data['Churn Value']
data = data.drop(["Churn Value"], axis=1)
data["Churn Value"] = temp


zip_agg = data.groupby('Zip Code').agg(
    customer_count=('Churn Value', 'size'),
    churn_sum=('Churn Value', 'sum')
).reset_index()

zip_agg['Zip_Churn_Rate'] = zip_agg['churn_sum'] / zip_agg['customer_count']

rate_map = zip_agg.set_index('Zip Code')['Zip_Churn_Rate'].to_dict()

data['Zip_Churn_Rate'] = data['Zip Code'].map(rate_map)

data = data.drop(columns=['Zip Code'])

temp = data['Zip_Churn_Rate']
data = data.drop(["Zip_Churn_Rate"], axis=1)
data.insert(1, "Zip_Churn_Rate", temp)


data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')


data = data.drop(['Tech Support_No internet service','Streaming TV_No internet service', 'Streaming Movies_No internet service', 'Gender_Male', 
                  'Device Protection_Yes','Online Backup_Yes','Payment Method_Mailed check', 'Payment Method_Credit card (automatic)',
                  'Multiple Lines_Yes', 'Senior Citizen_Yes','Device Protection_No internet service','Streaming TV_Yes'], axis=1)


temp = data['Churn Value']
data = data.drop(["Churn Value"], axis=1)
data['Churn Value'] = temp




x_value = data.drop(["Churn Value"], axis=1)
y_value = data['Churn Value']



x_train,x_test,y_train,y_test = train_test_split(x_value, y_value, test_size=0.1, stratify=y_value, random_state=42)



params = params = {
    'scale_pos_weight': 1.1,

    'n_estimators': 1500,
    'learning_rate': 0.005,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'n_jobs': -1,

    'max_depth': 4,                          
    'min_child_weight': 12,                   
    'gamma': 0.2,                            
    'subsample': 0.9,
    'colsample_bytree': 0.9,

    'reg_alpha': 0.005,
    'reg_lambda': 1,

    'random_state': 42
}


model = xgb.XGBClassifier(**params)

model.fit(x_train, y_train)

prediction = model.predict(x_test)
report_dict = classification_report(y_test, prediction, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.iloc[:-1, :3]  # drop 'accuracy', keep precision/recall/f1

plt.figure(figsize=(6,4))
sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f')
plt.title("Classification Report Heatmap")
plt.ylabel("Classes")
plt.xlabel("Metrics")
plt.tight_layout()
plt.savefig("classification_report_heatmap.png", dpi=300)
plt.show()

RocCurveDisplay.from_estimator(model, x_test, y_test)
plt.plot([0,1],[0,1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()


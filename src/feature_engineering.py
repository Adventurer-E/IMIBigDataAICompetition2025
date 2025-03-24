import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Define dtype mapping for available columns
dtypes = {
    'customer_id': str,
    'debit_credit': str,
    'transaction_type': str,
    'industry_code': str,
    'employee_count': float,
    'sales': float,
    'industry': str,
}
file_path = "final_data_before_engineered.csv"  # Update with your file's path
df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)

# Convert 'amount_cad' to numeric (if not already) and then to absolute values
df['amount_cad'] = pd.to_numeric(df['amount_cad'], errors='coerce').abs()

# Convert 'transaction_date' to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

# Check if required columns exist before processing
required_columns = ['customer_id', 'amount_cad', 'transaction_date', 'debit_credit']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in the dataset: {missing_columns}")

# Behavioral Features: Count transactions per customer
behavioral_features = df.groupby('customer_id')['amount_cad'].count().reset_index()
behavioral_features.rename(columns={'amount_cad': 'transaction_count'}, inplace=True)

# Median and variance of transaction amounts by customer and debit_credit
amount_stats = df.groupby(['customer_id', 'debit_credit'])['amount_cad'].agg(['median', 'var']).unstack(fill_value=0)
amount_stats.columns = ['_'.join(col).strip() for col in amount_stats.columns]
amount_stats.reset_index(inplace=True)

# Time-Series Features: Rolling Aggregations over different windows
def rolling_aggregation(data, customer_id, date_col, value_col, windows):
    data = data.sort_values(by=[customer_id, date_col])
    for window in windows:
        data[f"{value_col}_rolling_{window}_days"] = (
            data.groupby(customer_id)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )
    return data

windows = [30, 60, 90]
df = rolling_aggregation(df, 'customer_id', 'transaction_date', 'amount_cad', windows)

# Combine Engineered Features
engineered_df = df.merge(behavioral_features, on='customer_id', how='left')
engineered_df = engineered_df.merge(amount_stats, on='customer_id', how='left')

# Save the engineered features to CSV
output_path = "engineered_features.csv"
engineered_df.to_csv(output_path, index=False)
print(f"Feature engineering complete. File saved to {output_path}.")

df = engineered_df
df.drop(columns=['trxn_country', 'trxn_province', 'trxn_city', 'kyc_country', 'kyc_province', 'kyc_city',
         'established_date', 'onboard_date', 'industry_code', 'source', 'card_merchant_category'], inplace=True, errors='ignore')
to_be_grouped= ['industry']
df = pd.get_dummies(df, columns=['transaction_type', 'debit_credit'], drop_first=True)


def read_json_to_dict(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

file_path = "industry_to_general_industry.json"
industry_to_general_industry = read_json_to_dict(file_path)

if industry_to_general_industry:
    print(f"Successfully loaded JSON data. Number of entries: {len(industry_to_general_industry)}")

for index, row in df.iterrows():
    if row['industry'] in industry_to_general_industry.keys():
        df.at[index, 'general_industry'] = industry_to_general_industry[row['industry']]
    else:
        print(row['industry'])

df.drop(columns=['industry'],inplace=True)
df = pd.get_dummies(df, columns=['general_industry'], drop_first=True)
df.drop(columns=['distance_trxn_kyc'], inplace=True, errors='ignore')
df['amount_cad_rolling_90_days'] = df['amount_cad_rolling_90_days'] - df['amount_cad_rolling_60_days']
df['amount_cad_rolling_60_days'] = df['amount_cad_rolling_60_days'] - df['amount_cad_rolling_30_days']
df.to_csv("engineered_numerized.csv", index=False)

transactions = df
kyc = pd.read_csv("kyc.csv")
kyc['industry_code'] = kyc['industry_code'].fillna(-1).replace("other", -1).astype(int)
kyc.drop(columns=['country', 'province', 'city', 'established_date', 'onboard_date'], inplace=True, errors='ignore')
kyc_industry_codes = pd.read_csv("/content/drive/MyDrive/BigDataAICompetition/data/kyc_industry_codes.csv")
kyc = kyc.merge(kyc_industry_codes, on="industry_code", how="left")
kyc.drop(columns=['industry_code'], inplace=True, errors='ignore')
new_features = ['trans_frequency', 'most_fre_trans_method', 'std_trans_amount']
direct_copy_features = ['amount_cad_rolling_30_days',
       'amount_cad_rolling_60_days', 'amount_cad_rolling_90_days',
       'transaction_count', 'median_credit', 'median_debit', 'var_credit',
       'var_debit']
industry_dummies = ['general_industry_Construction',
       'general_industry_Education & Healthcare',
       'general_industry_Finance & Real Estate',
       'general_industry_Hospitality & Entertainment',
       'general_industry_Manufacturing', 'general_industry_Mining & Energy',
       'general_industry_Others',
       'general_industry_Personal & Community Services',
       'general_industry_Professional & Business Services',
       'general_industry_Telecommunications & Utilities',
       'general_industry_Transportation & Logistics',
       'general_industry_Wholesale & Retail Trade']
transaction_type_dummies_ = ['transaction_type_card',
       'transaction_type_cheque', 'transaction_type_eft',
       'transaction_type_emt', 'transaction_type_wire']
transaction_types = transaction_type_dummies_ + ['transaction_type_abm']
for index, row in kyc.iterrows():
    customer_id = row['customer_id']
    customer_transactions = transactions[transactions['customer_id'] == customer_id]
    if customer_transactions.empty:
        for new_feature in new_features + direct_copy_features:
            kyc.at[index, new_feature] = None
        continue
    for industry_dummy in industry_dummies:
        kyc.at[index, industry_dummy] = customer_transactions.iloc[0][industry_dummy]
    # for merchant_dummy in merchant_dummies:
    #     kyc.at[index, merchant_dummy] = customer_transactions[merchant_dummy].sum()
    customer_transactions['transaction_date'] = pd.to_datetime(customer_transactions['transaction_date'], errors='coerce')
    customer_transactions.sort_values(by='transaction_date', inplace=True)
    customer_transactions.reset_index(drop=True, inplace=True)
    timeframe = (customer_transactions.iloc[-1]['transaction_date'] - customer_transactions.iloc[0]['transaction_date']).days
    if timeframe == 0:
        timeframe = 1
    kyc.at[index, 'trans_frequency'] = len(customer_transactions) / timeframe * 100

    for index_, row_ in customer_transactions.iterrows():
        customer_transactions.at[index, 'transaction_type_abm'] = (1 if row_['transaction_type_card'] == 0 and row_['transaction_type_cheque'] == 0 and row_['transaction_type_eft'] == 0 and row_['transaction_type_emt'] == 0 and row_['transaction_type_wire'] == 0 else 0)
    kyc.at[index, 'most_fre_trans_method'] = customer_transactions[transaction_types].sum().idxmax()

    kyc.at[index, 'std_trans_amount'] = customer_transactions['amount_cad'].std()
    if timeframe == 1:
        kyc.at[index, 'avg_time_btw_trans'] = 0
    else:
        kyc.at[index, 'avg_time_btw_trans'] = (customer_transactions.iloc[1:]['transaction_date'] - customer_transactions.iloc[:-1]['transaction_date']).mean().days
    customer_industry = [industry_dummy for industry_dummy in industry_dummies if kyc.at[index, industry_dummy] == 1]
    if customer_industry != []:
        customer_industry = customer_industry[0][17:]
    else:
        customer_industry = 'Agriculture & Forestry'
    for feature in direct_copy_features:
        kyc.at[index, feature] = customer_transactions.iloc[0][feature]
# Deal with empty values
for col in ['employee_count', 'sales', 'var_credit', 'var_debit', 'trans_frequency', 'std_trans_amount', 'amount_cad_rolling_30_days', 'median_credit', 'median_debit']:
    kyc[col].fillna(0, inplace=True)
for col in industry_dummies:
    if col == "general_industry_Others":
        kyc[col].fillna(True, inplace=True)
    else:
        kyc[col].fillna(False, inplace=True)
kyc = pd.get_dummies(kyc, columns=['most_fre_trans_method'], drop_first=True)
kyc.drop(columns=['industry','merchant_dist_index','transaction_count','amount_cad_rolling_60_days', 'amount_cad_rolling_90_days', 'avg_time_btw_trans'], inplace=True, errors='ignore')
kyc.to_csv("kyc_full.csv", index=False)
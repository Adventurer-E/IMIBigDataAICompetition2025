import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_dir = Path('/mnt/data')
output_dir = Path('/mnt/output/task1')
# List of file names
filenames = ['kyc.csv', 'kyc_industry_codes.csv', 'abm.csv', 'card.csv', 'cheque.csv', 'eft.csv', 'emt.csv', 'wire.csv']

# Initialize an empty list to store the DataFrames
dataframes = []

# Read each file and store it in a list
for filename in filenames:
    try:
        print(data_dir / filename)
        df = pd.read_csv(data_dir / filename)
        dataframes.append(df)
    except FileNotFoundError:
        print(f"Error: File {filename} not found. Please check the file path.")
    except pd.errors.ParserError:
        print(f"Error: File {filename} could not be parsed. Please check its format.")
    except Exception as e:
        print(f"An unexpected error occurred while reading {filename}: {e}")


#   define each dataframe in dataframes with variable names

# Define each DataFrame with a variable name
kyc = dataframes[0]
kyc_industry_codes = dataframes[1]
abm = dataframes[2]
card = dataframes[3]
cheque = dataframes[4]
eft = dataframes[5]
emt = dataframes[6]
wire = dataframes[7]

transaction_dataset = [abm, card, eft, emt, wire, cheque]
kyc_dataset = [kyc, kyc_industry_codes]

# Visulize transaction datasets
dataset_names = ['abm', 'card', 'eft', 'emt', 'wire', 'cheque']  # Names of the datasets

for i, (name, df) in enumerate(zip(dataset_names, transaction_dataset)):
    # Look at missing values
    missing_values = df.isnull().sum()
    print(f"Missing values in dataset '{name}':")
    print(missing_values)
    print("\n")  # Add spacing for clarity in the output

# Define a dictionary to map merchant category codes (MCCs) to their corresponding descriptions.
merchant_data = {
    "merchant_category": [
        "other", 5542, 5814, 5411, 5541, 5812, 4816, 5251, 5734, 5200,
        5912, 4121, 5310, 9399, 4812, 7399, 7311, 5968, 4215, 7523,
        5921, 5499, 5300, 4814, 4899, 5943, 5818, 5999, 5085, 5511,
        7372, 5732, 5691, 5815, 5817, 6300, 7538, 8398, 5533, 5045,
        5942, 5047, 7011, 5211, 5816, 8699, 5331, 5651, 4900, 5712,
        7542, 5311, 4722, 4784, 5941, 5039, 5655, 8099
    ],
    "merchant_description": [
        "Miscellaneous services", "Automated Fuel Dispensers", "Fast Food Restaurants",
        "Grocery Stores, Supermarkets", "Service Stations (with or without ancillary services)",
        "Eating Places, Restaurants", "Computer Network/Information Services", "Hardware Stores",
        "Computer Software Stores", "Home Supply Warehouse Stores", "Drug Stores, Pharmacies",
        "Taxicabs and Limousines", "Discount Stores", "Government Services (Not Elsewhere Classified)",
        "Telecommunication Equipment and Telephone Sales", "Business Services (Not Elsewhere Classified)",
        "Advertising Services", "Direct Marketing – Continuity/Subscription Merchants",
        "Courier Services – Air and Ground, Freight Forwarders", "Automobile Parking Lots and Garages",
        "Package Stores – Beer, Wine, and Liquor", "Miscellaneous Food Stores", "Wholesale Clubs",
        "Telecommunication Services, including Local and Long Distance Calls, Credit Card Calls, etc.",
        "Cable, Satellite, and Other Pay Television and Radio Services",
        "Stationery, Office Supplies, Printing, and Writing Paper", "Internet Gambling",
        "Miscellaneous and Specialty Retail Stores", "Industrial Supplies (Not Elsewhere Classified)",
        "Automobile and Truck Dealers (New and Used) Sales, Service, Repairs, Parts, and Leasing",
        "Computer Programming, Data Processing, and Integrated Systems Design Services",
        "Electronics Stores", "Men's and Women's Clothing Stores", "Digital Goods – Media, Books, Movies, Music",
        "Digital Goods – Games", "Insurance Sales, Underwriting, and Premiums",
        "Automotive Repair Shops (Non-Dealer)", "Charitable and Social Service Organizations",
        "Automotive Parts and Accessories Stores", "Computers, Peripherals, and Software", "Book Stores",
        "Medical, Dental, Ophthalmic, and Hospital Equipment and Supplies",
        "Lodging – Hotels, Motels, Resorts, Central Reservation Services (Not Elsewhere Classified)",
        "Lumber and Building Materials Stores", "Digital Goods – Applications (Excluding Games)",
        "Membership Organizations (Not Elsewhere Classified)", "Variety Stores", "Family Clothing Stores",
        "Utilities – Electric, Gas, Water, and Sanitary", "Furniture, Home Furnishings, and Equipment Stores",
        "Car Washes", "Department Stores", "Travel Agencies and Tour Operators", "Tolls and Bridge Fees",
        "Sporting Goods Stores", "Construction Materials (Not Elsewhere Classified)",
        "Sports and Riding Apparel Stores", "Health Practitioners, Medical Services (Not Elsewhere Classified)"
    ]
}

# Create a DataFrame for merchant categories, counts, and descriptions
merchant_category_df = pd.DataFrame(merchant_data)

# Join the merchant_description with df card on merchant_category column

# Convert 'merchant_category' columns to the same data type for merging
card['merchant_category'] = card['merchant_category'].astype('string')
merchant_category_df['merchant_category'] = merchant_category_df['merchant_category'].astype('string')

# Merge the DataFrames based on the 'merchant_category' column
card = pd.merge(card, merchant_category_df, on='merchant_category', how='left')

kyc_dataset = [kyc, kyc_industry_codes]
transaction_dataset = [abm, card, eft, emt, wire, cheque]

# Convert all column that is country, province, city to string
for df in transaction_dataset + kyc_dataset:
    for col in df.columns:
        if col in ['country', 'province', 'city']:
            df[col] = df[col].astype('string')

# Convert 'established_date' and 'onboard_date' in 'kyc' to datetime
for col in ['established_date', 'onboard_date']:
    if col in kyc.columns:
        kyc[col] = pd.to_datetime(kyc[col], errors='coerce')

# Change industry column in kyc_industry_code to string
kyc_industry_codes['industry'] = kyc_industry_codes['industry'].astype('string')

# Convert 'transaction_date' and 'transaction_time' to datetime in transaction datasets
for df in transaction_dataset:
    if 'transaction_date' in df.columns:  # Only process if 'transaction_date' exists
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(
            df['transaction_time'], format='%H:%M:%S', errors='coerce'
        )

# Convert 'debit_credit' columns to string type in all transaction datasets
for df in transaction_dataset:
    if 'debit_credit' in df.columns:
        df['debit_credit'] = df['debit_credit'].astype('string')

# Change all customer_id columns to string type in all relevant DataFrames
for df in transaction_dataset:
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].astype('string')

# Change merchant_category and merchant_description to string
if 'merchant_category' in card.columns:
    card['merchant_category'] = card['merchant_category'].astype('string')
if 'merchant_description' in card.columns:
    card['merchant_description'] = card['merchant_description'].astype('string')

# Check all df info

# Assuming all necessary libraries and dataframes are loaded as in the provided code.

# Function to display info for all dataframes in a list
def display_df_info(dfs, df_names):
  for df, name in zip(dfs, df_names):
    print(f"DataFrame: {name}")
    display(df.info())
    print("-" * 40)

# Define a list of your dataframes and their names
dataframes_to_check = [kyc, abm, card, eft, emt, wire, cheque, kyc_industry_codes]
dataframe_names = ['kyc','abm', 'card', 'eft', 'emt', 'wire', 'cheque', 'kyc_industry_codes']

display_df_info(dataframes_to_check, dataframe_names)

# Replace `industry_code` of rows where it is 'other', '9999', or NaN with 9999
kyc.loc[
    (kyc['industry_code'] == 'other') | (kyc['industry_code'] == 9999) | (kyc['industry_code'].isnull()),
    'industry_code'
] = 9999

# Convert the `industry_code` to integer for consistency
kyc['industry_code'] = kyc['industry_code'].astype(int)

kyc.industry_code.value_counts()
# Now, it matches the number of rows in kyc_industry_code, the 2 df are ready to be marged

# Perform a left join between `kyc` and `kyc_industry_codes` on `industry_code`
kyc_full = kyc.merge(kyc_industry_codes, how='left', on='industry_code')
kyc_full['industry'] = kyc_full['industry'].astype('string')

# prompt: For all the df with the country, province, city, identify the df

# Identify DataFrames with 'country', 'province', and 'city' columns
location_dfs = []
for df_name, df in zip(filenames, dataframes):
    if all(col in df.columns for col in ['country', 'province', 'city']):
        location_dfs.append((df_name, df))

# Display information about the identified DataFrames
for df_name, df in location_dfs:
  print(f"DataFrame '{df_name}' has 'country', 'province', and 'city' columns:")
  display(df.info())

# Df kyc, abm, card has country, province, and city columns.

# Fill country, province, city 'other'

# Fill NaN values in 'country', 'province', and 'city' columns with 'other' for kyc, abm, and card DataFrames
for df in [kyc_full, abm, card]:
    for col in ['country', 'province', 'city']:
        if col in df.columns:
            df[col] = df[col].fillna('other')


# if province = 'other', but city is not 'other', then map the province to known city (when province is not other and when city is not other)

def map_province_to_city(df):
    """Maps province to known city when province is 'other' but city is not."""

    # Create a mapping from city to province where province is not 'other'
    city_province_mapping = df[df['province'] != 'other'].groupby('city')['province'].first().to_dict()

    # Apply the mapping to rows where province is 'other' and city is not 'other'
    df.loc[(df['province'] == 'other') & (df['city'] != 'other'), 'province'] = df.loc[(df['province'] == 'other') & (df['city'] != 'other'), 'city'].map(city_province_mapping).fillna('other')
    return df

kyc_full = map_province_to_city(kyc_full)

# Now, check all the country -> province -> city as a hierachy

# Assuming kyc_full DataFrame is already created as in the provided code.

def display_hierarchy(df):
    """Displays the country -> province -> city hierarchy."""
    for country in df['country'].unique():
        print(f"Country: {country}")
        country_df = df[df['country'] == country]
        for province in country_df['province'].unique():
            print(f"  Province: {province}")
            province_df = country_df[country_df['province'] == province]
            for city in province_df['city'].unique():
                print(f"    City: {city}")
        print("-" * 20)


def map_province_to_city(df):
    """Maps province to known city when province is 'other' but city is not."""
    # Create a mapping from city to province where province is not 'other'
    city_province_mapping = df[df['province'] != 'other'].groupby('city')['province'].first().to_dict()

    # Map province values based on city, where province is 'other' and city is valid
    df.loc[(df['province'] == 'other') & (df['city'] != 'other'), 'province'] = (
        df.loc[(df['province'] == 'other') & (df['city'] != 'other'), 'city']
        .map(city_province_mapping)
        .fillna('other')
    )
    return df

# Apply the mapping function
kyc_full = map_province_to_city(kyc_full)

# Display the refined hierarchy
display_hierarchy(kyc_full)


# Standardize city names
def standardize_city_names(city):
    city_mapping = {
        'FORT ST JOHN': 'FORT ST. JOHN',
        'ST CATHARINES': 'ST. CATHARINES',
        'ST ALBERT': 'ST. ALBERT',
    }
    # Check if city is a valid string before calling upper()
    if pd.notna(city) and isinstance(city, str):
        return city_mapping.get(city.upper(), city)
    else:
        return city  # Return the original value if it's not a string or pd.NA

kyc_full['city'] = kyc_full['city'].apply(standardize_city_names)

# Display the refined hierarchy
display_hierarchy(kyc_full)

abm = map_province_to_city(abm)

city_mapping = {
    'STCATHARINES': 'ST. CATHARINES',
    'HALTONHLLS': 'HALTON HILLS',
    'NORTHYORK': 'NORTH YORK',
    'STTHOMAS': 'ST. THOMAS',
    'NIAGARA FLS': 'NIAGARA FALLS',
    'STJEROME': 'ST. JEROME',
    'COTESTLUC': 'COTE SAINT-LUC',
    'COLEHARBOUR': 'COLE HARBOUR',
    'STJOHNS': 'ST. JOHN\'S',
}
abm['city'] = abm['city'].replace(city_mapping)

display_hierarchy(abm)

# Correct the province for ciies
card.loc[(card['country'] == 'CA') & (card['city'] == 'LINDSAY') & (card['province'] == 'other'), 'province'] = 'ON'
card.loc[(card['country'] == 'GB') & (card['city'] == 'LONDON') & (card['province'] == 'other'), 'province'] = 'England'

city_mapping = {
    'GRANDE PRAI': 'GRANDE PRAIRIE',
    'GRANDE PRAIRI': 'GRANDE PRAIRIE',
    'NIAGARA FAL': 'NIAGARA FALLS',
    'ST CATHARINES': 'ST. CATHARINES',
    'ST CATHARIN': 'ST. CATHARINES',
    'RICHMOND HI': 'RICHMOND HILL',
    'PETERBOROUG': 'PETERBOROUGH',
}
card['city'] = card['city'].replace(city_mapping)
card['province'] = card['province'].replace({'PQ': 'QC'})

card.loc[(card['province'] == 'other') & (card['city'] == 'TORONTO'), 'province'] = 'ON'
card.loc[(card['province'] == 'other') & (card['city'] == 'VANCOUVER'), 'province'] = 'BC'
card.loc[(card['province'] == 'other') & (card['city'] == 'VANCOUVER'), 'province'] = 'BC'

# Replace invalid or misplaced entries with 'other'
card.loc[(df['province'] == 'NS') & (card['city'] == 'CANADA'), 'city'] = 'other'
card.loc[(df['country'] == 'GB') & (card['province'] == 'ON'), 'province'] = 'other'

# Change all country, province, city columns to string

# Assuming dataframes list is already defined and populated as in the provided code.

# Iterate through all dataframes
for df in dataframes:
    for col in df.columns:
        if col in ['country', 'province', 'city']:
            # Check if the column exists before attempting to convert
            if col in df:
                df[col] = df[col].astype('string')

# Check all debit_credit columns
dataset_names = ['abm', 'card', 'eft', 'emt', 'wire', 'cheque']
transaction_dataset = [abm, card, eft, emt, wire, cheque]

# For all debit card credit card column, if it is not "debit" "credit" but "C" and "D" instead, change them to "credit" "debit"

# Iterate through transaction datasets and fix 'debit_credit' values
for df in transaction_dataset:
    if 'debit_credit' in df.columns:
        df['debit_credit'] = df['debit_credit'].astype('string')  # Ensure it's string type
        df['debit_credit'] = df['debit_credit'].replace({'C': 'credit', 'D': 'debit'})

# card.amount_cad make a new column, capture absoluate value, and a new column: card_value_negative_original?

# Create 'card_amount_cad_abs' column with absolute values
card['card_amount_cad_abs'] = card['amount_cad'].abs()

# Create 'card_value_negative_original' column indicating negative original values
card['card_value_negative_original'] = card['amount_cad'] < 0


# kyc_full.employee_count.fillna(0, inplace=True)
# kyc_full.sales.fillna(0, inplace=True)

kyc_full.employee_count.fillna(0, inplace=True)
kyc_full.sales.fillna(0, inplace=True)

# Convert all column that is country, province, city to string
for df in transaction_dataset or kyc_dataset:
    for col in df.columns:
        if col in ['country', 'province', 'city']:
            df[col] = df[col].astype('string')

# Convert 'established_date' and 'onboard_date' in 'kyc' to datetime
for col in ['established_date', 'onboard_date']:
    if col in kyc.columns:
        kyc[col] = pd.to_datetime(kyc[col], errors='coerce')

# Change industry column in kyc_industry_code to string
kyc_industry_codes['industry'] = kyc_industry_codes['industry'].astype('string')

# Convert 'transaction_date' and 'transaction_time' to datetime in transaction datasets
for df in transaction_dataset:
    if 'transaction_date' in df.columns:  # Only process if 'transaction_date' exists
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(
            df['transaction_time'], format='%H:%M:%S', errors='coerce'
        )

# Convert 'debit_credit' columns to string type in all transaction datasets
for df in transaction_dataset:
    if 'debit_credit' in df.columns:
        df['debit_credit'] = df['debit_credit'].astype('string')

# Change all customer_id columns to string type in all relevant DataFrames
for df in transaction_dataset:
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].astype('string')

# Change merchant_category and merchant_description to string
if 'merchant_category' in card.columns:
    card['merchant_category'] = card['merchant_category'].astype('string')
if 'merchant_description' in card.columns:
    card['merchant_description'] = card['merchant_description'].astype('string')



## Join Datasets
abm.drop(columns=['abm_id'], inplace=True)
card.drop(columns=['card_trxn_id'], inplace=True)
eft.drop(columns=['eft_id'], inplace=True)
emt.drop(columns=['emt_id'], inplace=True)
wire.drop(columns=['wire_id'], inplace=True)
cheque.drop(columns=['cheque_id'], inplace=True)
# Assign transaction types
def assigntype(df_name):
    # Define the mapping of transaction type based on the DataFrame
    type_map = {
        "cheque": "cheque",
        "emt": "emt",
        "eft": "eft",
        "wire": "wire",
        "abm": "abm",
        "card": "card"
    }
    return type_map.get(df_name, "Unknown")

# List of DataFrames with their names
transaction_dfs = {
    "cheque": cheque,
    "emt": emt,
    "eft": eft,
    "wire": wire,
    "abm": abm,
    "card": card
}

# Add the transaction_type column to each DataFrame
for name, df in transaction_dfs.items():
    df["transaction_type"] = assigntype(name)

# Combine cheque, emt, eft, and wire
transaction_dfs_1 = pd.concat([cheque, emt, eft, wire], ignore_index=True)

# Add prefix to location columns
kyc_full = kyc_full.rename(columns={
    "country": "kyc_country",
    "province": "kyc_province",
    "city": "kyc_city"
})

# Join transaction_dfs_1 with kyc_full
transaction_dfs_1_kyc_full = transaction_dfs_1.merge(kyc_full, on='customer_id', how='left')

# Prefix unique columns in abm
abm = abm.rename(columns={
    "cash_indicator": "abm_cash_indicator",
    "country": "trxn_country",
    "province": "trxn_province",
    "city": "trxn_city"
})

# Prefix unique columns in card
card = card.rename(columns={
    "merchant_category": "card_merchant_category",
    "merchant_description": "card_merchant_description",
    "ecommerce_ind": "card_ecommerce_ind",
    "country": "trxn_country",
    "province": "trxn_province",
    "city": "trxn_city"
})

# Join abm and card with kyc_full respectively
abm_kyc_full = abm.merge(kyc_full, on='customer_id', how='left')
card_kyc_full = card.merge(kyc_full, on='customer_id', how='left')

abm_kyc_full_droped = abm_kyc_full.drop(['trxn_country', 'trxn_province', 'trxn_city'], axis=1)

final_df = pd.concat([transaction_dfs_1_kyc_full, abm_kyc_full_droped], ignore_index=True)
card_kyc_full_droped = card_kyc_full.drop(['trxn_country', 'trxn_province', 'trxn_city', 'card_merchant_category', 'card_ecommerce_ind', 'card_merchant_description', 'card_amount_cad_abs', 'card_value_negative_original'], axis=1)
final_df = pd.concat([final_df, card_kyc_full_droped], ignore_index=True)

final_df = final_df.drop(['established_date', 'onboard_date', 'transaction_time', 'kyc_country', 'kyc_province', 'kyc_city'], axis=1)

final_df.to_csv(output_dir / 'final_data_before_engineered.csv', index=False)
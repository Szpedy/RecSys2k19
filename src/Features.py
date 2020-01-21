MODE = 'test'
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools

## SOURCES
## github.com/mustelideos/recsys-challenge-2019/blob/0e093e341d8e3c728693fc54260432fe4f879709/scripts/011_Features_Items.py
## github.com/logicai-io/recsys2019/blob/c0caed220056d3758d7e8b0032e89429fc07f8cf/src/recsys/data_prep/extract_hotel_dense_features.py

RATING_MAP = {"Satisfactory Rating": 1, "Good Rating": 2, "Very Good Rating": 3, "Excellent Rating": 4}
STAR_MAP = {"1 Star": 1, "2 Star": 2, "3 Star": 3, "4 Star": 4, "5 Star": 5}
HOTEL_CAT = {
    "Hotel": "hotel",
    "Resort": "resort",
    "Hostal (ES)": "hostal",
    "Motel": "motel",
    "House / Apartment": "house",
}
IMPORTANT_FEATURES = [
    "Free WiFi (Combined)",
    "Swimming Pool (Combined Filter)",
    "Car Park",
    "Serviced Apartment",
    "Air Conditioning",
    "Spa (Wellness Facility)",
    "Pet Friendly",
    "All Inclusive (Upon Inquiry)",
]


def normalize_feature_name(name):
    return name.replace(" ", "_").lower()


def densify(d, properties):
    values = [None] * properties.shape[0]
    for i, p in enumerate(properties):
        for k in d:
            if k in p:
                values[i] = d[k]
    return values


def get_features_from_metadata():
    # Read metadata
    df_items = pd.read_csv('../recSysData/item_metadata.csv')
    df_items["properties"] = df_items["properties"].str.split("|").map(set)
    df_items["rating"] = densify(RATING_MAP, df_items["properties"])
    df_items["stars"] = densify(STAR_MAP, df_items["properties"])
    df_items["hotel_cat"] = densify(HOTEL_CAT, df_items["properties"])

    for f in IMPORTANT_FEATURES:
        df_items[normalize_feature_name(f)] = df_items["properties"].map(lambda p: f in p).astype(np.int)

    df_items = df_items.drop(columns=["properties"])
    df_items['impressions'] = df_items['item_id']
    df_items = df_items.drop(columns=['item_id'])

    df_items.to_csv('../data/item_metadata_dense.csv', index=False)
platforms = {
    'AA': 0,
    'AE': 1,
    'AR': 2,
    'AT': 3,
    'AU': 4,
    'BE': 5,
    'BG': 6,
    'BR': 7,
    'CA': 8,
    'CH': 9,
    'CL': 10,
    'CN': 11,
    'CO': 12,
    'CZ': 13,
    'DE': 14,
    'DK': 15,
    'EC': 16,
    'ES': 17,
    'FI': 18,
    'FR': 19,
    'GR': 20,
    'HK': 21,
    'HR': 22,
    'HU': 23,
    'ID': 24,
    'IE': 25,
    'IL': 26,
    'IN': 27,
    'IT': 28,
    'JP': 29,
    'KR': 30,
    'MX': 31,
    'MY': 32,
    'NL': 33,
    'NO': 34,
    'NZ': 35,
    'PE': 36,
    'PH': 37,
    'PL': 38,
    'PT': 39,
    'RO': 40,
    'RS': 41,
    'RU': 42,
    'SE': 43,
    'SG': 44,
    'SI': 45,
    'SK': 46,
    'TH': 47,
    'TR': 48,
    'TW': 49,
    'UK': 50,
    'US': 51,
    'UY': 52,
    'VN': 53,
    'ZA': 54,
}

countries = {
    ' Albania': 0,
    ' Algeria': 1,
    ' Andorra': 2,
    ' Angola': 3,
    ' Antigua and Barbuda': 4,
    ' Argentina': 5,
    ' Armenia': 6,
    ' Aruba': 7,
    ' Australia': 8,
    ' Austria': 9,
    ' Azerbaijan': 10,
    ' Bahamas': 11,
    ' Bahrain': 12,
    ' Bangladesh': 13,
    ' Barbados': 14,
    ' Belarus': 15,
    ' Belgium': 16,
    ' Belize': 17,
    ' Benin': 18,
    ' Bermudas': 19,
    ' Bolivia': 20,
    ' Bosnia and Herzegovina': 21,
    ' Botswana': 22,
    ' Brazil': 23,
    ' Brunei': 24,
    ' Bulgaria': 25,
    ' Burkina Faso': 26,
    ' Burundi': 27,
    ' Cambodia': 28,
    ' Cameroon': 29,
    ' Canada': 30,
    ' Cape Verde': 31,
    ' Cayman Islands': 32,
    ' Chile': 33,
    ' China': 34,
    ' Colombia': 35,
    ' Cook Islands': 36,
    ' Costa Rica': 37,
    ' Crimea': 38,
    ' Croatia': 39,
    ' Cuba': 40,
    ' Curacao': 41,
    ' Cyprus': 42,
    ' Czech Republic': 43,
    ' Denmark': 44,
    ' Dominica': 45,
    ' Dominican Republic': 46,
    ' Ecuador': 47,
    ' Egypt': 48,
    ' El Salvador': 49,
    ' Estonia': 50,
    ' Ethiopia': 51,
    ' Faroe Islands': 52,
    ' Fiji': 53,
    ' Finland': 54,
    ' France': 55,
    ' French Antilles': 56,
    ' French Guiana': 57,
    ' French Polynesia': 58,
    ' Georgia': 59,
    ' Germany': 60,
    ' Ghana': 61,
    ' Gibraltar': 62,
    ' Greece': 63,
    ' Greenland': 64,
    ' Grenada': 65,
    ' Guam': 66,
    ' Guatemala': 67,
    ' Haiti': 68,
    ' Honduras': 69,
    ' Hong Kong': 70,
    ' Hungary': 71,
    ' Iceland': 72,
    ' India': 73,
    ' Indonesia': 74,
    ' Iran': 75,
    ' Iraq': 76,
    ' Ireland': 77,
    ' Israel': 78,
    ' Italy': 79,
    ' Ivory Coast': 80,
    ' Jamaica': 81,
    ' Japan': 82,
    ' Jordan': 83,
    ' Kazakhstan': 84,
    ' Kenya': 85,
    ' Kosovo': 86,
    ' Kuwait': 87,
    ' Laos': 88,
    ' Latvia': 89,
    ' Lebanon': 90,
    ' Lesotho': 91,
    ' Liechtenstein': 92,
    ' Lithuania': 93,
    ' Luxembourg': 94,
    ' Madagascar': 95,
    ' Malawi': 96,
    ' Malaysia': 97,
    ' Maldives': 98,
    ' Malta': 99,
    ' Mauritania': 100,
    ' Mauritius': 101,
    ' Mexico': 102,
    ' Moldova': 103,
    ' Monaco': 104,
    ' Montenegro': 105,
    ' Morocco': 106,
    ' Mozambique': 107,
    ' Myanmar': 108,
    ' Namibia': 109,
    ' Nepal': 110,
    ' Netherlands': 111,
    ' New Caledonia': 112,
    ' New Zealand': 113,
    ' Nicaragua': 114,
    ' Niger': 115,
    ' Nigeria': 116,
    ' Northern Mariana Islands': 117,
    ' Norway': 118,
    ' Oman': 119,
    ' Pakistan': 120,
    ' Palestinian Territories': 121,
    ' Panama': 122,
    ' Papua New Guinea': 123,
    ' Paraguay': 124,
    ' Peru': 125,
    ' Philippines': 126,
    ' Poland': 127,
    ' Portugal': 128,
    ' Puerto Rico': 129,
    ' Qatar': 130,
    ' Republic of Macedonia': 131,
    ' Romania': 132,
    ' Russia': 133,
    ' Réunion': 134,
    ' Saint Lucia': 135,
    ' Saint Vincent and the Grenadines': 136,
    ' Samoa': 137,
    ' Saudi Arabia': 138,
    ' Senegal': 139,
    ' Serbia': 140,
    ' Seychelles': 141,
    ' Singapore': 142,
    ' Sint Maarten': 143,
    ' Slovakia': 144,
    ' Slovenia': 145,
    ' South Africa': 146,
    ' South Korea': 147,
    ' Spain': 148,
    ' Sri Lanka': 149,
    ' Sudan': 150,
    ' Suriname': 151,
    ' Swaziland': 152,
    ' Sweden': 153,
    ' Switzerland': 154,
    ' São Tomé and Príncipe': 155,
    ' Taiwan': 156,
    ' Tanzania': 157,
    ' Thailand': 158,
    ' The Gambia': 159,
    ' Togo': 160,
    ' Tonga': 161,
    ' Trinidad and Tobago': 162,
    ' Tunisia': 163,
    ' Turkey': 164,
    ' Turks and Caicos Islands': 165,
    ' US Virgin Islands': 166,
    ' USA': 167,
    ' Uganda': 168,
    ' Ukraine': 169,
    ' United Arab Emirates': 170,
    ' United Kingdom': 171,
    ' Uruguay': 172,
    ' Uzbekistan': 173,
    ' Vanuatu': 174,
    ' Vatican City': 175,
    ' Venezuela': 176,
    ' Vietnam': 177,
    ' Zambia': 178,
    ' Zimbabwe': 179,
}

devices = {'mobile': 0,
           'desktop': 1,
           'tablet': 2}


def add_features_to_df(df):
    df.device = df.device.map(devices)
    df.platform = df.platform.map(platforms)
    df['country'] = df.city.apply(lambda x: x.split(',')[-1]).map(countries)
    df['mean_prices'] = df.prices / df.prices.mean()
    df['median_prices'] = df.prices / df.prices.median()
    df = df.merge(df_items, on='impressions', how='left')
    return df


if __name__ == '__main__':
    # read items metadata
    get_features_from_metadata()
    df_items = pd.read_csv('../data/item_metadata_dense.csv')

    # train
    df_train = pd.read_csv('../data/train_exploded.csv')
    df_train = add_features_to_df(df_train)
    df_train.dropna()
    df_train.to_csv('../data/train_new_features.csv', index=False)

    # test
    df_test = pd.read_csv('../data/test_exploded.csv')
    df_test = add_features_to_df(df_test)
    df_test.to_csv('../data/test_new_features.csv', index=False)

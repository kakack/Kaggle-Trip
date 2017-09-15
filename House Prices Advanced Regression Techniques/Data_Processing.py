# Imports
import pandas as pd
import numpy as np

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#njobs = 4

def data_basic_processing(data):
    try:
        data.drop("Id", axis = 1, inplace = True)
    except AttributeError as e:
        pass
    # Handle missing values for features where median/mean or most common value doesn't make sense

    # Alley : data description says NA means "no alley access"
    data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna("None")
    # BedroomAbvGr : NA most likely means 0
    data.loc[:, "BedroomAbvGr"] = data.loc[:, "BedroomAbvGr"].fillna(0)
    # BsmtQual etc : data description says NA for basement features is "no basement"
    data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna("No")
    data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna("No")
    data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("No")
    data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("No")
    data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("No")
    data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
    data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtHalfBath"].fillna(0)
    data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)
    # CentralAir : NA most likely means No
    data.loc[:, "CentralAir"] = data.loc[:, "CentralAir"].fillna("N")
    # Condition : NA most likely means Normal
    data.loc[:, "Condition1"] = data.loc[:, "Condition1"].fillna("Norm")
    data.loc[:, "Condition2"] = data.loc[:, "Condition2"].fillna("Norm")
    # EnclosedPorch : NA most likely means no enclosed porch
    data.loc[:, "EnclosedPorch"] = data.loc[:, "EnclosedPorch"].fillna(0)
    # External stuff : NA most likely means average
    data.loc[:, "ExterCond"] = data.loc[:, "ExterCond"].fillna("TA")
    data.loc[:, "ExterQual"] = data.loc[:, "ExterQual"].fillna("TA")
    # Fence : data description says NA means "no fence"
    data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna("No")
    # FireplaceQu : data description says NA means "no fireplace"
    data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna("No")
    data.loc[:, "Fireplaces"] = data.loc[:, "Fireplaces"].fillna(0)
    # Functional : data description says NA means typical
    data.loc[:, "Functional"] = data.loc[:, "Functional"].fillna("Typ")
    # GarageType etc : data description says NA for garage features is "no garage"
    data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("No")
    data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("No")
    data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("No")
    data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("No")
    data.loc[:, "GarageArea"] = data.loc[:, "GarageArea"].fillna(0)
    data.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
    # HalfBath : NA most likely means no half baths above grade
    data.loc[:, "HalfBath"] = data.loc[:, "HalfBath"].fillna(0)
    # HeatingQC : NA most likely means typical
    data.loc[:, "HeatingQC"] = data.loc[:, "HeatingQC"].fillna("TA")
    # KitchenAbvGr : NA most likely means 0
    data.loc[:, "KitchenAbvGr"] = data.loc[:, "KitchenAbvGr"].fillna(0)
    # KitchenQual : NA most likely means typical
    data.loc[:, "KitchenQual"] = data.loc[:, "KitchenQual"].fillna("TA")
    # LotFrontage : NA most likely means no lot frontage
    data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)
    # LotShape : NA most likely means regular
    data.loc[:, "LotShape"] = data.loc[:, "LotShape"].fillna("Reg")
    # MasVnrType : NA most likely means no veneer
    data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna("None")
    data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)
    # MiscFeature : data description says NA means "no misc feature"
    data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna("No")
    data.loc[:, "MiscVal"] = data.loc[:, "MiscVal"].fillna(0)
    # OpenPorchSF : NA most likely means no open porch
    data.loc[:, "OpenPorchSF"] = data.loc[:, "OpenPorchSF"].fillna(0)
    # PavedDrive : NA most likely means not paved
    data.loc[:, "PavedDrive"] = data.loc[:, "PavedDrive"].fillna("N")
    # PoolQC : data description says NA means "no pool"
    data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna("No")
    data.loc[:, "PoolArea"] = data.loc[:, "PoolArea"].fillna(0)
    # SaleCondition : NA most likely means normal sale
    data.loc[:, "SaleCondition"] = data.loc[:, "SaleCondition"].fillna("Normal")
    # ScreenPorch : NA most likely means no screen porch
    data.loc[:, "ScreenPorch"] = data.loc[:, "ScreenPorch"].fillna(0)
    # TotRmsAbvGrd : NA most likely means 0
    data.loc[:, "TotRmsAbvGrd"] = data.loc[:, "TotRmsAbvGrd"].fillna(0)
    # Utilities : NA most likely means all public utilities
    data.loc[:, "Utilities"] = data.loc[:, "Utilities"].fillna("AllPub")
    # WoodDeckSF : NA most likely means no wood deck
    data.loc[:, "WoodDeckSF"] = data.loc[:, "WoodDeckSF"].fillna(0)

    # Some numerical features are actually really categories
    data = data.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                                        50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                                        80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                                        150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                         "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                         })


    data = data.replace({"Alley": {"Grvl": 1, "Pave": 2},
                         "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
                         "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                          "ALQ": 5, "GLQ": 6},
                         "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                                          "ALQ": 5, "GLQ": 6},
                         "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                                        "Min2": 6, "Min1": 7, "Typ": 8},
                         "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                         "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                         "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
                         "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                         "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                         "Street": {"Grvl": 1, "Pave": 2},
                         "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}}
                        )

    data["SimplOverallQual"] = data.OverallQual.replace({1: 1, 2: 1, 3: 1,  # bad
                                                         4: 2, 5: 2, 6: 2,  # average
                                                         7: 3, 8: 3, 9: 3, 10: 3  # good
                                                         })
    data["SimplOverallCond"] = data.OverallCond.replace({1: 1, 2: 1, 3: 1,  # bad
                                                         4: 2, 5: 2, 6: 2,  # average
                                                         7: 3, 8: 3, 9: 3, 10: 3  # good
                                                         })
    data["SimplPoolQC"] = data.PoolQC.replace({1: 1, 2: 1,  # average
                                               3: 2, 4: 2  # good
                                               })
    data["SimplGarageCond"] = data.GarageCond.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
    data["SimplGarageQual"] = data.GarageQual.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
    data["SimplFireplaceQu"] = data.FireplaceQu.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
    data["SimplFireplaceQu"] = data.FireplaceQu.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
    data["SimplFunctional"] = data.Functional.replace({1: 1, 2: 1,  # bad
                                                       3: 2, 4: 2,  # major
                                                       5: 3, 6: 3, 7: 3,  # minor
                                                       8: 4  # typical
                                                       })
    data["SimplKitchenQual"] = data.KitchenQual.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
    data["SimplHeatingQC"] = data.HeatingQC.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    data["SimplBsmtFinType1"] = data.BsmtFinType1.replace({1: 1,  # unfinished
                                                           2: 1, 3: 1,  # rec room
                                                           4: 2, 5: 2, 6: 2  # living quarters
                                                           })
    data["SimplBsmtFinType2"] = data.BsmtFinType2.replace({1: 1,  # unfinished
                                                           2: 1, 3: 1,  # rec room
                                                           4: 2, 5: 2, 6: 2  # living quarters
                                                           })
    data["SimplBsmtCond"] = data.BsmtCond.replace({1: 1,  # bad
                                                   2: 1, 3: 1,  # average
                                                   4: 2, 5: 2  # good
                                                   })
    data["SimplBsmtQual"] = data.BsmtQual.replace({1: 1,  # bad
                                                   2: 1, 3: 1,  # average
                                                   4: 2, 5: 2  # good
                                                   })
    data["SimplExterCond"] = data.ExterCond.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    data["SimplExterQual"] = data.ExterQual.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })

    # 2* Combinations of existing features
    # Overall quality of the house
    data["OverallGrade"] = data["OverallQual"] * data["OverallCond"]
    # Overall quality of the garage
    data["GarageGrade"] = data["GarageQual"] * data["GarageCond"]
    # Overall quality of the exterior
    data["ExterGrade"] = data["ExterQual"] * data["ExterCond"]
    # Overall kitchen score
    data["KitchenScore"] = data["KitchenAbvGr"] * data["KitchenQual"]
    # Overall fireplace score
    data["FireplaceScore"] = data["Fireplaces"] * data["FireplaceQu"]
    # Overall garage score
    data["GarageScore"] = data["GarageArea"] * data["GarageQual"]
    # Overall pool score
    data["PoolScore"] = data["PoolArea"] * data["PoolQC"]
    # Simplified overall quality of the house
    data["SimplOverallGrade"] = data["SimplOverallQual"] * data["SimplOverallCond"]
    # Simplified overall quality of the exterior
    data["SimplExterGrade"] = data["SimplExterQual"] * data["SimplExterCond"]
    # Simplified overall pool score
    data["SimplPoolScore"] = data["PoolArea"] * data["SimplPoolQC"]
    # Simplified overall garage score
    data["SimplGarageScore"] = data["GarageArea"] * data["SimplGarageQual"]
    # Simplified overall fireplace score
    data["SimplFireplaceScore"] = data["Fireplaces"] * data["SimplFireplaceQu"]
    # Simplified overall kitchen score
    data["SimplKitchenScore"] = data["KitchenAbvGr"] * data["SimplKitchenQual"]
    # Total number of bathrooms
    data["TotalBath"] = data["BsmtFullBath"] + (0.5 * data["BsmtHalfBath"]) + \
                        data["FullBath"] + (0.5 * data["HalfBath"])
    # Total SF for house (incl. basement)
    data["AllSF"] = data["GrLivArea"] + data["TotalBsmtSF"]
    # Total SF for 1st + 2nd floors
    data["AllFlrsSF"] = data["1stFlrSF"] + data["2ndFlrSF"]
    # Total SF for porch
    data["AllPorchSF"] = data["OpenPorchSF"] + data["EnclosedPorch"] + \
                         data["3SsnPorch"] + data["ScreenPorch"]
    # Has masonry veneer or not
    data["HasMasVnr"] = data.MasVnrType.replace({"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                                 "Stone": 1, "None": 0})
    # House completed before sale or not
    data["BoughtOffPlan"] = data.SaleCondition.replace({"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                                        "Family": 0, "Normal": 0, "Partial": 1})

    # 3* Polynomials on the top 10 existing features
    data["OverallQual-s2"] = data["OverallQual"] ** 2
    data["OverallQual-s3"] = data["OverallQual"] ** 3
    data["OverallQual-Sq"] = np.sqrt(data["OverallQual"])
    data["AllSF-2"] = data["AllSF"] ** 2
    data["AllSF-3"] = data["AllSF"] ** 3
    data["AllSF-Sq"] = np.sqrt(data["AllSF"])
    data["AllFlrsSF-2"] = data["AllFlrsSF"] ** 2
    data["AllFlrsSF-3"] = data["AllFlrsSF"] ** 3
    data["AllFlrsSF-Sq"] = np.sqrt(data["AllFlrsSF"])
    data["GrLivArea-2"] = data["GrLivArea"] ** 2
    data["GrLivArea-3"] = data["GrLivArea"] ** 3
    data["GrLivArea-Sq"] = np.sqrt(data["GrLivArea"])
    data["SimplOverallQual-s2"] = data["SimplOverallQual"] ** 2
    data["SimplOverallQual-s3"] = data["SimplOverallQual"] ** 3
    data["SimplOverallQual-Sq"] = np.sqrt(data["SimplOverallQual"])
    data["ExterQual-2"] = data["ExterQual"] ** 2
    data["ExterQual-3"] = data["ExterQual"] ** 3
    data["ExterQual-Sq"] = np.sqrt(data["ExterQual"])
    data["GarageCars-2"] = data["GarageCars"] ** 2
    data["GarageCars-3"] = data["GarageCars"] ** 3
    data["GarageCars-Sq"] = np.sqrt(data["GarageCars"])
    data["TotalBath-2"] = data["TotalBath"] ** 2
    data["TotalBath-3"] = data["TotalBath"] ** 3
    data["TotalBath-Sq"] = np.sqrt(data["TotalBath"])
    data["KitchenQual-2"] = data["KitchenQual"] ** 2
    data["KitchenQual-3"] = data["KitchenQual"] ** 3
    data["KitchenQual-Sq"] = np.sqrt(data["KitchenQual"])
    data["GarageScore-2"] = data["GarageScore"] ** 2
    data["GarageScore-3"] = data["GarageScore"] ** 3
    data["GarageScore-Sq"] = np.sqrt(data["GarageScore"])

def dif_num_and_cat(data):
    categorical_features = data.select_dtypes(include=["object"]).columns
    numerical_features = data.select_dtypes(exclude=["object"]).columns
    try:
        numerical_features = numerical_features.drop("SalePrice")
    except AttributeError as e:
        pass
    print("Numerical features : " + str(len(numerical_features)))
    print("Categorical features : " + str(len(categorical_features)))
    data_num = data[numerical_features]
    data_cat = data[categorical_features]

    print("NAs for numerical features in train : " + str(data_num.isnull().values.sum()))
    data_num = data_num.fillna(data_num.median())
    print("Remaining NAs for numerical features in train : " + str(data_num.isnull().values.sum()))

    return data, data_num, data_cat
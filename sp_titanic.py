# Spaceship Titanic
# Datasets used and inspiration for this project are from the following Kaggle competition: https://www.kaggle.com/competitions/spaceship-titanic
# Used for non-commercial, educational purposes under the competition rules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

def initial_eda(df):
    """ Plots various graphs for exploratory data analysis """
    df_plot = df.copy()
    df_plot['Transported'] = df_plot['Transported'].map({True:'Transported', False:'Not transported'})
    plt.figure()
    sns.countplot(x='Transported', data=df_plot)
    plt.title('Transported vs Not Transported')
    plt.ylabel('No. of Passengers')
    plt.tight_layout()
    plt.savefig('figures/transported_counts.png')
    plt.close()
    df_edited = df.copy()
    for col in ('HomePlanet', 'Destination', 'CryoSleep', 'VIP'):
            if col in ('HomePlanet', 'Destination'):
                order = df_edited.groupby(col)['Transported'].mean().sort_values().index
            else:
                order = None
                if col == 'CryoSleep':
                    df_edited[col] = df_edited[col].map({True:'CryoSleep', False:'No CryoSleep'})
                else:
                    df_edited[col] = df_edited[col].map({True:'VIP', False:'Not VIP'})
            plt.figure()
            sns.barplot(x=col, y='Transported', data=df_edited, estimator=np.mean, order=order)
            plt.title('Transported Rate by ' + col)
            plt.xlabel(col)
            plt.ylabel('Proportion Transported')
            plt.tight_layout()
            plt.savefig('figures/transported_rate_by_' + col + '.png')
            plt.close()
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure()
    sns.barplot(x=missing.index, y=missing.values)
    plt.title('Missing values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Proportion of Values Missing')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figures/missing_values.png')
    plt.close()
    df_cabin_split = split_cabin_col(df).copy()
    df_cabin_split['Side'] = df_cabin_split['Side'].map({'P':'Port', 'S':'Starboard'})
    for col in ('Deck', 'Side'):
        if col == 'Deck':
            not_small = df_cabin_split['Deck'].value_counts()[lambda x : x >= 10].index
            df_new = df_cabin_split[df_cabin_split['Deck'].isin(not_small)]
            order = sorted(not_small)
        else:
            order = None
            df_new = df_cabin_split.copy()
        plt.figure()
        sns.barplot(x=col, y='Transported', data=df_new, estimator=np.mean, order=order)
        plt.title('Transported Rate by ' + col)
        plt.xlabel(col)
        plt.ylabel('Proportion Transported')
        plt.tight_layout()
        plt.savefig('figures/transported_rate_by_' + col + '.png')
        plt.close()

def split_cabin_col(df):
    """ Splits the 'Cabin' columns into 'Deck', 'Cabin_num' and 'Side' components """
    cabin_split = df['Cabin'].str.split('/', expand=True)
    cabin_split.columns = ['Deck', 'Cabin_num', 'Side']
    cabin_split['Cabin_num'] = pd.to_numeric(cabin_split['Cabin_num'], errors='coerce') # converts strings to numbers
    cabin_index = df.columns.get_loc('Cabin')
    return pd.concat([df.iloc[:, :cabin_index], cabin_split, df.iloc[:, cabin_index + 1:]], axis=1) # replaces 'Cabin' column with the three new ones

def impute_with_mode(df, col):
    """ Imputes missing values with the mode for that column """
    mode = df[col].mode()[0]
    df.loc[df[col].isna(), col] = mode

def impute_with_median(df, col):
    """ Imputes missing values with the median for that column """
    median = df[col].median()
    df.loc[df[col].isna(), col] = median

def impute_spends(df, col):
    """ Imputes missing values for the 'money spent' columns depending on 'CryoSleep' """
    df.loc[(df[col].isna()) & (df['CryoSleep'] == True), col] = 0
    df.loc[(df[col].isna()) & (df['CryoSleep'] == False), col] = median_spends[col]

def bin_variable(df, col, parts=5):
    """ Bins numerical variables in-place """
    df[col] = pd.qcut(df[col], q=parts, duplicates='drop')
    return df

def clean_data(df):
    """ Multi-step data cleaning """
    df = df.drop(columns=['PassengerId', 'Name'])
    df = split_cabin_col(df)

    # now we impute missing values

    for col in ('HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP'):
        impute_with_mode(df, col)
    for col in ('Cabin_num', 'Age'):
        impute_with_median(df, col)
    for col in spending:
        impute_spends(df, col)

    for col in ('CryoSleep', 'VIP'):
        df[col] = df[col].map({True:1, False:0})
    df['Side'] = df['Side'].map({'S':1, 'P':0})
    df = df.rename(columns={'Side':'Starboard'})
    df.insert(df.columns.get_loc('VRDeck') + 1, 'TotalSpend', df[spending].sum(axis=1))
    for col in ('Cabin_num', 'Age'):
        df = bin_variable(df, col)
    df = pd.get_dummies(df, columns=['HomePlanet', 'Deck', 'Cabin_num', 'Destination', 'Age']) # replaces categorical variables with boolean dummy variables
    return df

def repeated_cross_validation(model, X, y):
    """ Repeated stratified 5-fold cross-validation for the random forest model """
    cross_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    return cross_val_score(model, X, y, cv=cross_val, scoring='accuracy')

def log_regression(X, y):
    """ Multi-step process for implementing and testing logistic regression model """

    model = LogisticRegression(max_iter=5000)

    # hyperparameter tuning

    parameters = {'C':[0.01, 0.1, 1, 10, 100, 1000]}
    cross_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    grid_search = GridSearchCV(model, parameters, scoring='accuracy', n_jobs=-1, cv=cross_val)
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)
    grid_search.fit(X_scaled, y)
    results = pd.DataFrame(grid_search.cv_results_)
    summary = results[['param_C', 'mean_test_score', 'std_test_score']]
    summary = summary.rename(columns={'param_C':'hyperparameter', 'mean_test_score':'mean', 'std_test_score':'std'})
    print()
    print('Hyperparameter tuning: ')
    print()
    print(summary.sort_values('mean', ascending=False).to_string(index=False))
    print()
    print('C=10 marginally better than C=1')
    print()
    best_model = grid_search.best_estimator_
    best_index = grid_search.best_index_
    mean = grid_search.best_score_
    std = grid_search.cv_results_['std_test_score'][best_index]
    print('Cross-validation accuracy per model:')
    print()
    print(f"Logistic Regression: {100*mean:.0f} ± {100*std:.0f}")

    # plots logistic regression coefficients for feature comparison

    coefs = pd.Series(best_model.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
    plt.figure(figsize=(20, max(6, 0.25 * len(coefs.index))))
    sns.barplot(x=coefs, y=coefs.index)
    plt.title('Logistic Regression Coefficients of Each Feature')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.savefig('figures/lr_coefficients.png')

def random_forest(X, y, X_test):
    """ Multi-step process for implementing and testing random forest model, with final predictions for submission """
    model = RandomForestClassifier(random_state=seed, n_jobs=-1)
    scores = repeated_cross_validation(model, X, y)
    print(f"Random Forest: {100*scores.mean():.0f} ± {100*scores.std():.0f}")
    model.fit(X, y)

    # plots 'feature importances' to assess the influence that each feature had on the model

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(20, max(6, 0.25 * len(importances.index))))
    sns.barplot(x=importances, y=importances.index)
    plt.title('Relative Importance of Each Feature in Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('figures/feature_importances.png')

    # uses random forest model to obtain predictions for test data

    predictions = model.predict(X_test)
    predictions = predictions == 1
    submission = pd.DataFrame({'PassengerId':test_id, 'Transported':predictions})
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':

    seed = 345
    np.random.seed(seed)
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    initial_eda(train)

    test_id = test['PassengerId']
    train['Transported'] = train['Transported'].map({True:1, False:0})
    spending = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    median_spends = {}
    for col in spending:
        median_spends[col] = train.loc[(train[col].notna()) & (train['CryoSleep'] == False), col].median()

    # apply the data cleaning process

    train = clean_data(train)
    test = clean_data(test)

    # format data to prepare for applying machine learning models

    X = train.drop(columns=['Transported']).copy()
    y = train['Transported'].copy()
    X_test = test.copy()
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
    feature_names = X.columns

    # finally, apply both machine learning models

    log_regression(X, y)
    random_forest(X, y, X_test)
"""Binary classification of Spaceship Titanic passengers using logistic regression and random forest."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

def initial_eda(df):
    """ Plots various graphs for exploratory data analysis """
    df_edited = df.copy()
    df_edited['CryoSleep'] = df_edited['CryoSleep'].map({True:'CryoSleep', False:'No CryoSleep'})
    for col in ('HomePlanet', 'Destination', 'CryoSleep'):
        if col in ('HomePlanet', 'Destination'):
            order = df_edited.groupby(col)['Transported'].mean().sort_values().index
        else:
            order = None
        plt.figure()
        sns.barplot(x=col, y='Transported', data=df_edited, estimator=np.mean, order=order)
        plt.title('Transported Rate by ' + col)
        plt.xlabel(col)
        plt.ylabel('Proportion Transported')
        plt.tight_layout()
        plt.savefig('outputs/figures/transported_rate_by_' + col + '.png')
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
        plt.savefig('outputs/figures/transported_rate_by_' + col + '.png')
        plt.close()

def split_cabin_col(df):
    """ Splits the 'Cabin' columns into 'Deck', 'Cabin_num' and 'Side' components """
    cabin_split = df['Cabin'].str.split('/', expand=True)
    cabin_split.columns = ['Deck', 'Cabin_num', 'Side']
    cabin_split['Cabin_num'] = pd.to_numeric(cabin_split['Cabin_num'], errors='coerce')
    cabin_index = df.columns.get_loc('Cabin')
    return pd.concat([df.iloc[:, :cabin_index], cabin_split, df.iloc[:, cabin_index + 1:]], axis=1) # replaces 'Cabin' column with the three new ones

def impute_with_mode(df, col, mode):
    """ Imputes missing values with a pre-computed mode for that column """
    df.loc[df[col].isna(), col] = mode

def impute_with_median(df, col, median):
    """ Imputes missing values with a pre-computed median for that column """
    df.loc[df[col].isna(), col] = median

def impute_spends(df, col, median_spends):
    """ Imputes missing values for the 'money spent' columns depending on 'CryoSleep' """
    df.loc[(df[col].isna()) & (df['CryoSleep'] == True), col] = 0
    df.loc[(df[col].isna()) & (df['CryoSleep'] == False), col] = median_spends[col]

def bin_variable(df, col, bin_edges):
    """ Bins a numerical variable using pre-computed bin edges """
    df[col] = pd.cut(df[col], bins=bin_edges, include_lowest=True)
    return df

def clean_data(df, spending, median_spends, modes, medians, bins):
    """ Multi-step data cleaning """
    df = df.drop(columns=['PassengerId', 'Name'])
    df = split_cabin_col(df)
    for col in ('HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP'):
        impute_with_mode(df, col, modes[col])
    for col in ('Cabin_num', 'Age'):
        impute_with_median(df, col, medians[col])
    for col in spending:
        impute_spends(df, col, median_spends)
    for col in ('CryoSleep', 'VIP'):
        df[col] = df[col].map({True:1, False:0})
    df['Side'] = df['Side'].map({'S':1, 'P':0})
    df = df.rename(columns={'Side':'Starboard'})
    df.insert(df.columns.get_loc('VRDeck') + 1, 'TotalSpend', df[spending].sum(axis=1))
    for col in ('Cabin_num', 'Age'):
        df = bin_variable(df, col, bins[col])
    df = pd.get_dummies(df, columns=['HomePlanet', 'Deck', 'Cabin_num', 'Destination', 'Age'])
    return df


def log_regression(X, y, X_test, test_id, seed, feature_names):
    """ Multi-step process for implementing and testing logistic regression model, with final predictions for submission """

    model = LogisticRegression(max_iter=5000)

    # hyperparameter tuning

    parameters = {'C':[0.01, 0.1, 1, 10, 100, 1000]}
    cross_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    grid_search = GridSearchCV(model, parameters, scoring='accuracy', n_jobs=-1, cv=cross_val)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    mean = grid_search.best_score_
    std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    print(f"Logistic Regression: {100*mean:.0f} ± {100*std:.0f}")

    # plots logistic regression coefficients for feature comparison

    coefs = pd.Series(best_model.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
    plt.figure(figsize=(20, max(6, 0.25 * len(coefs.index))))
    sns.barplot(x=coefs, y=coefs.index)
    plt.title('Logistic Regression Coefficients of Each Feature')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.savefig('outputs/figures/lr_coefficients.png')
    plt.close()

    # uses logistic regression model to obtain predictions for test data

    X_test_scaled = scaler.transform(X_test)
    predictions = best_model.predict(X_test_scaled) == 1
    submission = pd.DataFrame({'PassengerId': test_id, 'Transported': predictions})
    submission.to_csv('outputs/lr_submission.csv', index=False)

def random_forest(X, y, X_test, seed, feature_names, test_id):
    """ Multi-step process for implementing and testing random forest model, with final predictions for submission """

    model = RandomForestClassifier(random_state=seed)

    # hyperparameter tuning

    parameters = {
        'n_estimators':     [100, 300],
        'max_depth':        [None, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    cross_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    grid_search = GridSearchCV(model, parameters, scoring='accuracy', n_jobs=-1, cv=cross_val)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    mean = grid_search.best_score_
    std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    print(f"Random Forest: {100*mean:.0f} ± {100*std:.0f}")

    # plots 'feature importances' to assess the influence that each feature had on the model

    importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(20, max(6, 0.25 * len(importances.index))))
    sns.barplot(x=importances, y=importances.index)
    plt.title('Relative Importance of Each Feature in Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('outputs/figures/feature_importances.png')
    plt.close()

    # uses random forest model to obtain predictions for test data

    predictions = best_model.predict(X_test) == 1
    submission = pd.DataFrame({'PassengerId': test_id, 'Transported': predictions})
    submission.to_csv('outputs/rf_submission.csv', index=False)


def main():

    seed = 345
    Path('outputs/figures').mkdir(parents=True, exist_ok=True)
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    initial_eda(train)
    test_id = test['PassengerId']
    train['Transported'] = train['Transported'].map({True:1, False:0})
    spending = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # compute all imputation values and bin edges from training data only
    
    train_split = split_cabin_col(train)
    modes = {col: train_split[col].mode()[0] for col in ('HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP')}
    medians = {col: train_split[col].median() for col in ('Cabin_num', 'Age')}
    median_spends = {col: train.loc[(train[col].notna()) & (train['CryoSleep'] == False), col].median() for col in spending}
    bins = {}

    for col in ('Cabin_num', 'Age'):
        _, bin_edges = pd.qcut(train_split[col].dropna(), q=5, retbins=True, duplicates='drop')
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf  # extend outer edges to cover all test values
        bins[col] = bin_edges

    train = clean_data(train, spending, median_spends, modes, medians, bins)
    test = clean_data(test, spending, median_spends, modes, medians, bins)
    X = train.drop(columns=['Transported']).copy()
    y = train['Transported'].copy()
    X_test = test.copy()
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
    feature_names = X.columns

    # finally, apply both machine learning models and compare cross-validation accuracy

    print('Cross-validation accuracy per model:')
    print()
    log_regression(X, y, X_test, test_id, seed, feature_names)
    random_forest(X, y, X_test, seed, feature_names, test_id)

if __name__ == '__main__':
    main()
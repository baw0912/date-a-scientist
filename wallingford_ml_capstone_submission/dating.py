#!/usr/bin/env python3
#
# Codecademy Machine Learning Fundamentals Capstone Project
# author: Ben Wallingford
# cohort: Feb 12, 2019
# submitted: Apr 7, 2019
#

import numpy as np
import pandas as pd
import re
import string

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


#
# say the person is a fluent english speaker if their 'speaks' column contains either:
#   - the string 'english (fluently'
#   - the string 'english' and no languages are marked as '(fluently)'
#
def detect_fluent_english(languages_spoken):
    if type(languages_spoken) != str:
        res = 0
    elif 'english (fluently)' in languages_spoken:
        res = 1
    elif 'english' in languages_spoken and '(fluently)' not in languages_spoken:
        res = 1
    else:
        res = 0
    return res


#
# say the person is fluent in something other than english if:
#   - it explicitly lists another language as fluent
#   - there are multiple languages, and none of the them are explicitly called fluent
#   - english is not listed as one of their languages (everyone is fluent in something)
#
def detect_fluent_not_english(languages_spoken):
    if type(languages_spoken) != str:
        res = 0
    elif re.search('(?!english) \(fluently\)', languages_spoken):
        res = 1
    elif ', ' in languages_spoken and '(fluently)' not in languages_spoken:
        res = 1
    elif 'english' not in languages_spoken:
        res = 1
    else:
        res = 0
    return res


def calculate_avg_word_length(in_string):
    num_words = in_string.count(' ') + 1
    just_letters = ''
    for c in in_string:
        if c in string.ascii_letters:
            just_letters += c

    return ( len(just_letters)/num_words )


def normalize_series(series_in):
    series_min = series_in.min()
    series_max = series_in.max()
    return series_in.apply(lambda x: ((x - series_min)/(series_max - series_min)))



class OKCupid:

    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    ### Functions for showing plots
    def show_drinking_income_scatter(self):
        plt.scatter(okc.df.drinks_code, okc.df.income, alpha=0.05)
        plt.xlabel('Drinking Frequency')
        plt.ylabel('Income')
        plt.title('Income vs. Drinking Frequency')
        plt.show()

        plt.scatter(okc.df.drinks_code, okc.df.income, alpha=0.05)
        plt.xlabel('Drinking Frequency')
        plt.ylabel('Income')
        plt.title('Income vs. Drinking Frequency')
        plt.ylim(0,300000)
        plt.show()
        return 0


    def show_age_histogram(self):
        plt.hist(self.df.age, bins=20)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.xlim(16,80)
        plt.title('Age Histogram')
        plt.show()
        return 0


    def show_income_histogram(self):
        plt.hist(self.df.income, bins=40)
        plt.xlabel('Income')
        plt.ylabel('Frequency')
        plt.xlim(0,200000)
        plt.title('Income Histogram')
        plt.show()
        return 0


    ### Functions for adding new columns to the DataFrame
    def add_state_country_column(self):
        self.df['state_country'] = self.df.location.transform(lambda x: x.split(', ')[1])
        return 0
    

    def add_fluent_english_column(self):
        self.df['fluent_english'] = self.df.speaks.apply(detect_fluent_english)
        return 0


    def add_fluent_not_english_column(self):
        self.df['fluent_not_english'] = self.df.speaks.apply(detect_fluent_not_english)
        return 0


    def add_drinks_code_column(self):
        drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
        self.df['drinks_code'] = self.df.drinks.map(drink_mapping)
        return 0


    def add_smokes_code_column(self):
        smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}
        self.df['smokes_code'] = self.df.smokes.map(smokes_mapping)
        return 0


    def add_drugs_code_column(self):
        drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
        self.df['drugs_code'] = self.df.drugs.map(drugs_mapping)
        return 0


    def add_sex_code_column(self):
        sex_mapping = {'m': 0, 'f': 1}
        self.df['sex_code'] = self.df.sex.map(sex_mapping)
        return 0


    def add_body_type_code_column(self):
        # this data isn't inherently ordered, but here is a go at it
        body_type_mapping = {
                'jacked': 0,
                'athletic': 1,
                'fit': 2,
                'average': 3,
                'thin': 4,
                'skinny': 5,
                #'rather not say': 6,   <-- don't map this to anything
                'a little extra': 7,
                'used up': 8,
                'curvy': 9,
                'full figured': 10,
                'overweight': 11
                }
        self.df['body_type_code'] = self.df.body_type.map(body_type_mapping)
        return 0


    def add_job_code_column(self):
        job_mapping = {
                'other': 0,
                'student': 1,
                'science / tech / engineering': 2,
                'computer / hardware / software': 3,
                'artistic / musical / writer': 4,
                'sales / marketing / biz dev': 5,
                'medicine / health': 6,
                'education / academia': 7,
                'executive / management': 8,
                'banking / financial / real estate': 9,
                'entertainment / media': 10,
                'law / legal services': 11,
                'hospitality / travel': 12,
                'construction / craftsmanship': 13,
                'clerical / administrative': 14,
                'political / government': 15,
                #'rather not say': 16, <-- don't map this anything
                'transportation': 17,
                'unemployed': 18,
                'retired': 19,
                'military': 20
                }
        self.df['job_code'] = self.df.job.map(job_mapping)
        return 0


    def get_all_essays(self):
        essay_cols = ["essay0", "essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9",]
        all_essays = self.df[essay_cols].replace(np.nan, '', regex=True)
        all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis = 1)
        return all_essays

    def add_essay_length_column(self):
        all_essays = self.get_all_essays()
        self.df['essay_len'] = all_essays.apply(lambda x: len(x))
        return 0


    def add_essay_avg_word_length_column(self):
        all_essays = self.get_all_essays()
        self.df['essay_avg_word_length'] = all_essays.apply(calculate_avg_word_length)
        return 0


    ### Fuctions to show basic summary info
    def show_basic_df_info(self):
        print(self.df.dtypes)


    def show_value_counts_for_columns(self, columns_to_show):
        for col in columns_to_show:
            print(f'\n### {col} ###\n') 
            print(self.df[col].value_counts())

        return 0


    ### Perform k-nearest neighbors modeling
    def k_nearest(self, label_key, cols, max_k):
        col_string = ', '.join(cols)
        print(f"\n### Can we determine {label_key} using k-nearest neighbors with columns: [{col_string}]?\n")

        n_df = pd.DataFrame()
        n_df[label_key] = self.df[label_key]
        for col in cols:
            n_df[col] = normalize_series(self.df[col])

        n_df.dropna(how='any', subset = cols, inplace=True)
        n_df.dropna(subset=[label_key], inplace=True)
        self.n_df = n_df

        training_data, validation_data, training_labels, validation_labels = train_test_split(n_df[cols], n_df[label_key], test_size=0.2, random_state=100)

        max_accuracy = 0
        best_k = 0
        for k in range(5,max_k, 5):
            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(training_data, training_labels)
            guesses = classifier.predict(validation_data)
            self.guesses = guesses
            self.validation_labels = validation_labels
            accuracy = accuracy_score(validation_labels, guesses)
            recall = recall_score(validation_labels, guesses, average='micro')
            precision = precision_score(validation_labels, guesses, average='micro')

            print(f'k={k} accuracy={accuracy} recall={recall} precision={precision}')

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_k = k

        print(f'Found best_k = {best_k} with accuracy of {max_accuracy}')

        return 0


    ### Perform support vector machine modeling
    def svm(self, label_key, cols, gamma, C):
        col_string = ', '.join(cols)
        print(f"\n### Can we determine {label_key} using SVM with columns: [{col_string}]?\n")

        n_df = pd.DataFrame()
        n_df[label_key] = self.df[label_key]
        for col in cols:
            n_df[col] = normalize_series(self.df[col])

        n_df.dropna(how='any', subset = cols, inplace=True)
        n_df.dropna(subset=[label_key], inplace=True)
        self.n_df = n_df

        training_data, validation_data, training_labels, validation_labels = train_test_split(n_df[cols], n_df[label_key], test_size=0.2, random_state=100)

        svc = SVC(kernel='rbf', gamma=gamma, C=C)
        svc.fit(training_data, training_labels)
        guesses = svc.predict(validation_data)
        accuracy = accuracy_score(validation_labels, guesses)
        recall = recall_score(validation_labels, guesses, average='micro')
        precision = precision_score(validation_labels, guesses, average='micro')
        print(f'gamma={gamma} C={C} accuracy={accuracy} recall={recall} precision={precision}')

        return 0


    ### Perform multiple regression to find english fluency
    def fluent_english_multiple_regression(self, cols):
        label_key = 'fluent_english'

        col_string = ', '.join(cols)
        print(f"\n### Can we determine {label_key} using Multiple Linear Regression with columns: [{col_string}]?\n")
        
        n_df = pd.DataFrame()
        n_df[label_key] = self.df[label_key]
        for col in cols:
            n_df[col] = normalize_series(self.df[col])

        n_df.dropna(how='any', subset = cols, inplace=True)
        n_df.dropna(subset=[label_key], inplace=True)
        self.n_df = n_df

        training_data, validation_data, training_labels, validation_labels = train_test_split(n_df[cols], n_df[label_key], test_size=0.2, random_state=100)
        
        regr = linear_model.LinearRegression()
        regr.fit(training_data, training_labels)
        r2 = regr.score(validation_data, validation_labels)

        print(f'R^2={r2}')

        return 0

    
    ### Perform k-nearest neighbors regression to find english fluency
    def fluent_english_k_neighbor_regression(self, cols):
        label_key = 'fluent_english'

        col_string = ', '.join(cols)
        print(f"\n### Can we determine {label_key} using K-Neighbors Regression with columns: [{col_string}]?\n")

        n_df = pd.DataFrame()
        n_df[label_key] = self.df[label_key]
        for col in cols:
            n_df[col] = normalize_series(self.df[col])

        n_df.dropna(how='any', subset = cols, inplace=True)
        n_df.dropna(subset=[label_key], inplace=True)
        self.n_df = n_df

        training_data, validation_data, training_labels, validation_labels = train_test_split(n_df[cols], n_df[label_key], test_size=0.2, random_state=100)

        for k in range(5,100,5):
            regr = KNeighborsRegressor(n_neighbors=k, weights='distance')
            regr.fit(training_data, training_labels)
            guesses = regr.predict(validation_data)
            r2 = regr.score(validation_data, validation_labels)
            print(f'k={k} R^2={r2}')

        return 0


if __name__ == '__main__':
    okc = OKCupid('profiles.csv')
    okc.show_basic_df_info()

    columns_to_count = [
            'body_type',
            'diet',
            'drinks',
            'drugs',
            'smokes',
            'education',
            'job',
            'offspring',
            'orientation',
            'pets',
            'religion',
            ]

    okc.show_value_counts_for_columns(columns_to_count)

    print("\n### Adding new columns")
    # add some columns based on interpretations of the data
    okc.add_state_country_column()
    okc.add_fluent_english_column()
    okc.add_fluent_not_english_column()

    # add a column or two that turns categories into numbers through .map()
    okc.add_drinks_code_column()
    okc.add_drugs_code_column()
    okc.add_smokes_code_column()
    okc.add_sex_code_column()
    okc.add_body_type_code_column()
    okc.add_job_code_column()

    # add essay word count column
    okc.add_essay_length_column()
    okc.add_essay_avg_word_length_column()

    print("### Displaying exploration plots")
    # some plots
    okc.show_drinking_income_scatter()
    okc.show_age_histogram()
    okc.show_income_histogram()

    ### classification question
    # Can we determine job with essay word counts, sex, and body_type
    cols = ['essay_len', 'sex_code', 'body_type_code']
    okc.k_nearest('job_code', cols, 100)
    okc.svm('job_code', cols, 3, 1)

    # Does it get better without body_type?
    cols = ['essay_len', 'sex_code']
    okc.k_nearest('job_code', cols, 100)
    okc.svm('job_code', cols, 3, 1)

    ## regression question
    # Can we tell if someone is a fluent english speaker, based on word length and age?
    cols = ['essay_avg_word_length', 'age']
    okc.fluent_english_multiple_regression(cols)
    okc.fluent_english_k_neighbor_regression(cols)


import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def beautify_treatment(treatment, treatment_dict):
    return treatment_dict[treatment]


def get_input_diagnosis():
    return {'diagnosis': input('whats your diagnosis? '),
            'age': int(input('whats your age? ')),
            'sex': input('whats your gender? f/m ')}


def main():
    treatment_df = pd.read_csv('people_medicine_data_large.csv')
    treatment_df = remove_useless_columns(treatment_df)

    common_treatment = set.intersection(set(treatment_df['treatment']))
    # common_diagnose = set.intersection(set(treatment_df['diagnosis']))
    # print(common_diagnose)
    treatment_dict = {treatment: i
                      for i, treatment in
                      zip(np.arange(len(common_treatment)),
                          common_treatment)}
    treatment_df = treatment_df.replace({'treatment': treatment_dict})

    x_data, labels = get_data_and_labels(treatment_df)
    x_train, x_test, y_train, y_test = train_test_split(x_data, labels,
                                                        test_size=0.33,
                                                        random_state=42)
    feature_columns_list, input_function, label_y = get_feature_columns_and_input_function(
        treatment_df, treatment_dict, x_train, y_train)
    model = get_trained_model(feature_columns_list, input_function, label_y,
                              treatment_dict, x_test, y_test)
    question_data = get_input_diagnosis()
    x_test_question = pd.DataFrame([question_data])
    prediction_function_question = \
        tf.estimator.inputs.pandas_input_fn(x=x_test_question,
                                            batch_size=
                                            x_test_question.shape[0],
                                            shuffle=False)
    predictions = list(model.predict(input_fn=prediction_function_question))
    res_treatment_dict = dict((v, k) for k, v in treatment_dict.items())
    predictions = [res_treatment_dict[pred['class_ids'][0]]
                   for pred in predictions]
    print(predictions)


def get_trained_model(feature_columns_list, input_function, label_y,
                      treatment_dict, x_test, y_test):
    print('values={} len={}'.format(treatment_dict.keys(), len(label_y)))
    model = tf.estimator.LinearClassifier(
        feature_columns=feature_columns_list, n_classes=len(label_y) + 1)
    model.train(input_fn=input_function, steps=5000)
    prediction_function = tf.estimator.inputs.pandas_input_fn(x=x_test,
                                                              batch_size=
                                                              x_test.shape[
                                                                  0],
                                                              shuffle=False)
    predictions = list(model.predict(input_fn=prediction_function))
    predictions = [pred['class_ids'][0] for pred in predictions]
    print(classification_report(y_test, predictions))
    return model


def get_feature_columns_and_input_function(treatment_df, treatment_dict,
                                           x_train, y_train):
    data_columns = treatment_df.columns.tolist()
    numeric_columns = ['age']
    data_columns = \
        list(set(data_columns) - set(numeric_columns + ['treatment']))
    numeric_features = {column: tf.feature_column.numeric_column(column) for
                        column in numeric_columns}
    categorical_features = {
        column: tf.feature_column.categorical_column_with_hash_bucket(
            column,
            hash_bucket_size=1000)
        for column in data_columns}
    numeric_features.update(categorical_features)
    feature_columns = numeric_features
    feature_columns[
        'sex'] = tf.feature_column.categorical_column_with_vocabulary_list(
        'sex', ['f', 'm'])
    age = tf.feature_column.bucketized_column(
        feature_columns['age'],
        boundaries=[0, 20, 40, 60, 80, 100])
    feature_columns['age'] = age
    input_function = tf.estimator.inputs.pandas_input_fn(x=x_train,
                                                         y=y_train,
                                                         batch_size=100,
                                                         num_epochs=1000,
                                                         shuffle=True)
    feature_columns_list = list(feature_columns.values())
    label_y = treatment_dict.keys()
    return feature_columns_list, input_function, label_y


def get_data_and_labels(treatment_df):
    x_data = treatment_df.drop(['treatment'], axis=1)
    labels = treatment_df['treatment']
    return x_data, labels


def remove_useless_columns(treatment_df):
    return treatment_df.drop(['person_instance', 'company'], axis=1)


if __name__ == '__main__':
    import logging

    logging.getLogger().setLevel(logging.INFO)
    main()

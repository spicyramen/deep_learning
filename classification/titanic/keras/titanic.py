# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import pandas as pd
import re

FILENAME = '../data/titanic.csv'


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def encode_one_hot(df, column, axis=1):
    return df.join(pd.get_dummies(df[column], column)).drop(column, axis=axis)


def extract_title(name):
    title = re.findall(r',(.*?)\.', name)[0].strip()

    if title in ['Dona', 'Lady', 'the Countess']:
        return 'Lady'
    elif title in ['Mme', 'Mlle']:
        return 'Mme'
    elif title in ['Capt', 'Don', 'Major', 'Sir', 'Jonkheer', 'Col']:
        return 'Sir'
    else:
        return title


def extract_deck(cabin):
    return cabin[0:1] if pd.notnull(cabin) else 'Unknown'


def build_model():
    m = Sequential([
        Dense(30, activation='relu', input_dim=feature_count, kernel_initializer='random_uniform'), # layer size = feature size + 1
        Dense(30, activation='relu'),
        Dense(1, activation='sigmoid'),  # TODO 1x sigmoid vs 2x softmax?
    ])
    m.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return m


EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 1337
NUM_FOLDS = 10

# Import data
# df_train = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')
dataset = pd.read_csv(FILENAME)
df_train = dataset.sample(frac=0.8, random_state=50)
df_test = dataset.drop(df_train.index)

# 19,0,3,"Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)",female,31,1,0,345763,18,,S

# Name           891 non-null object
# Sex            891 non-null object
# Ticket         891 non-null object
# Cabin          204 non-null object
# Embarked       889 non-null object
# PassengerId    891 non-null int64
# Pclass         891 non-null int64
# Age            714 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Fare           891 non-null float64
# Survived       891 non-null int64
# dtypes: float64(2), int64(5), object(5)

COLUMNS = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
           'Embarked']
CATEGORICAL = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
CONTINUOUS = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
LABEL = 'Survived'


df = pd.concat([df_train.drop('Survived', axis=1), df_test], ignore_index=True)  # type: pd.DataFrame
df = df.drop(['PassengerId', 'Ticket'], axis=1)  # type: pd.DataFrame

df['Title'] = df['Name'].map(extract_title)
df['Deck'] = df['Cabin'].map(extract_deck)
df['Age'] = normalize(df['Age'].fillna(df['Age'].median()))
df['Parch'] = normalize(df['Parch'])
df['SibSp'] = normalize(df['SibSp'])
df['Fare'] = normalize(df['Fare'])
df['FamSize'] = normalize(df['SibSp'] * df['Parch'])

df = encode_one_hot(df, 'Sex')
df = encode_one_hot(df, 'Embarked')
df = encode_one_hot(df, 'Deck')
df = encode_one_hot(df, 'Title')

x_all = df.drop(['Name', 'Cabin'], axis=1).as_matrix()

train_count = len(df_train)
feature_count = x_all.shape[1]
print('Number of features:', feature_count)

x_submit = x_all[train_count:]
x_train = x_all[:train_count]
y_train = df_train['Survived']

# Evaluate model using 10-fold cross-validation
model = KerasClassifier(build_fn=build_model, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=False)
cv = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = cross_val_score(model, x_train, y_train, cv=cv, n_jobs=-1)
print('Mean accuracy in %i-fold CV:\t' % NUM_FOLDS, results.mean())

# Build model on complete training data
model = build_model()
model.fit(x_train, y_train, nb_epoch=EPOCHS, batch_size=BATCH_SIZE, verbose=False)

# Evaluate model using confusion matrix
y_pred = model.predict_classes(x_train, verbose=False).flatten()
print('Final accuracy on training data:', accuracy_score(y_train, y_pred))
print(pd.crosstab(y_train, y_pred, rownames=['Real'], colnames=['Predicted'], margins=True))

# Store wrong predictions to file
row_filter = [y1 != y2 for (y1, y2) in zip(y_pred, y_train)]
df_wrong = df_train.copy()
df_wrong['SurvivedPrediction'] = y_pred
df_wrong = df_wrong[row_filter]
df_wrong.to_csv('wrong.csv', index=False)
print('Wrote', len(df_wrong), 'rows to wrong.csv')

# Submit
y_submit = model.predict_classes(x_submit, verbose=False).flatten()
df_submit = pd.DataFrame(y_submit, index=df_test['PassengerId'], columns=['Survived'])
df_submit.to_csv('submission.csv')
print('Wrote', len(df_submit), 'rows to submission.csv')

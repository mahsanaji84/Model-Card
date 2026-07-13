import sys

# includes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for fetching data
import requests
# Compile the PDF documents
import os

# to get the same result
np.random.seed(2)

################################
# Complete the functions below #
################################

# Download/create the dataset
def fetch():
  print("fetching dataset!")  # replace this with code to fetch the dataset
  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
  r = requests.get(url, stream=True)
  open('./adult.data', 'wb').write(r.content)


###################################Linear Regression Class#####################################
# The class LogisticRegression is inspired by the lab but I implemented it myself
class LogisticRegression:
    def __init__(self, w0, reg):
        self.w = np.array(w0, dtype=float)
        self.reg = reg
        self.best_reg = False
        self.best_w = False
        self.treshold = .5
        self.weight_lists = []

    def predict(self, X):
        dot = np.dot(X, self.w)
        sigmoid = 1.0 / (1 + np.exp(-dot))
        prediction = np.round(sigmoid)
        prediction =  np.where(prediction >= self.treshold , 1, -1)
        return prediction
    
    def test(self, X, y): 
        prediction = self.predict(X)
        return np.mean((prediction*y) < 0)

    def loss(self, X, y):
        """ This loss function penelize more when the model predicts label of who has more than 
          50K income wrongly. We did this because the dataset is imbalanced and the ratio of people
          with higher income to lower income is 1:3"""
        mask_greater = (y == 1)
        balance_coef = np.ones(len(y))
        balance_coef[mask_greater] = balance_coef[mask_greater] * 3
        return np.log(1 + np.exp(np.mean(-balance_coef*y* self.predict(X)))) + (self.reg/2) * np.sum(self.w**2)

    def gradient(self, X, y):
        # gradient of first term
        mask_greater = (y == 1)
        balance_coef = np.ones(len(y))
        balance_coef[mask_greater] = balance_coef[mask_greater] * 3
        numinator = (-balance_coef*y*X.T)*(np.exp(-balance_coef* y* self.predict(X)))
        denominator = (1 + np.exp((-balance_coef * y* self.predict(X))))
        return (numinator/denominator).mean(axis=1) + self.reg *self.w
    
    def train_report(self, training_loss, training_error):
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8,2))
        ax0.plot(training_loss)
        ax0.set_title('loss')
        ax1.plot(training_error)
        ax1.set_title('error rate')
        name = 'loss_error_reg_' + str(self.reg) +'.pdf'
        plt.savefig(name)
        # plt.show()

    def validation_report(self, test_x, test_y):
        print('The validation error is {:.2f}%'.format(self.test(test_x, test_y)*100))
        
    def test_report(self, test_x, test_y):
        print('The test error is {:.2f}%'.format(self.test(test_x, test_y)*100)) 

    def train(self, data, stepsize, n_steps, feature_subset, plot=False):
        X = data[feature_subset]
        y = np.array(data['is greater than 50'])
        losses = []
        errors = []
        for i in range(n_steps):
            # Gradient Descent
            self.w = self.w - stepsize * self.gradient(X, y)
            losses = losses + [self.loss(X, y)]
            errors = errors + [self.test(X, y)]
        print("Training completed: the train error is {:.2f}%".format(errors[-1]*100))
        self.train_report(np.array(losses), np.array(errors))
        return np.array(losses), np.array(errors)
    
    def tune_reg_term(self, train, validation_x, validation_y, cv_reg, stepsize=.2, n_steps=100, feature_subset=['bias'], plot=False):
        best_reg = cv_reg[0]
        best_error = 100
        self.weight_lists = []
        for reg in cv_reg:
            self.reg = reg
            self.train(train, stepsize, 100, feature_subset,  plot=plot)
            # print('for', reg, 'weight is:', self.w)
            error =  self.test(validation_x, validation_y) * 100
            if(error < best_error):
                best_error = error
                best_reg = reg
                self.best_w = self.w
            self.weight_lists.append(self.w)
            self.validation_report(validation_x, validation_y)
            mask = (validation_y == 1)
            print("for people with income greater than 50")
            self.validation_report(validation_x[mask], validation_y[mask])
        print(self.weight_lists)
        self.best_reg = best_reg
        return self.best_reg
    """ tune_treshold is commented because first I used it because my dataset is imbalanced; however, changing
      trshold ended up to very bad result. hence I changed the loss function instead."""
    """
    def tune_treshold(self, train, validation_x, validation_y, cv_treshold = [.2, .4, .5, .6, .7, .8], stepsize=.2, n_steps=100, feature_subset=features, plot=False):
        best_treshold = cv_treshold[0]
        best_error = 100
        for treshold in cv_treshold:
            self.treshold = treshold
            self.train(train, stepsize, 100, plot=plot)
            error =  model.test(validation_x, validation_y) * 100
            if(error < best_error):
                best_error = error
                best_reg = reg
                self.best_w = self.w
            print("validation errror is:")
            self.test_report(validation_x, validation_y)
            mask = (validation['is greater than 50'] == 1)
            print("validation errror for people with income greater than 50 is:")
            model.test_report(validation_x[mask], validation_y[mask])
        # self.treshold = treshold
        return self.treshold
    """
def prepare():
  ### read data ###
  columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
       'marital.status', 'occupation', 'relationship', 'race', 'sex',
       'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
       'income']
  df = pd.read_csv('./adult.data', names=columns, sep=', ', engine='python')

  ### data prepration ###
  # data cleaning
  df.drop('fnlwgt', axis=1, inplace= True)

  df['workclass'] = np.where(df['workclass'] == '?','not recorded', df['workclass'])
  df = pd.concat([df, pd.get_dummies(df['workclass'],prefix='workclass',prefix_sep=':')], axis=1)
  df.drop('workclass', axis=1, inplace= True)

  education_degree = {
      'Doctorate': 10,
      'Masters': 8,
      'Bachelors': 6,
      'Assoc-voc': 5,
      'Assoc-acdm': 5,
      'Some-college': 5,
      'Prof-school': 5,
      'HS-grad': 5,
      '12th': 4,
      '11th': 4,
      '10th': 3,
      '9th': 3,
      '7th-8th': 2,
      '5th-6th': 1,
      '1st-4th': 1,
      'Preschool': 0
  }
  df['education'] = df['education'].apply(lambda x: education_degree[x])

  df.drop('education.num', axis=1, inplace=True)

  marital_status = {
      'Married-civ-spouse': 'married',
      'Never-married': 'single',
      'Divorced': 'single',
      'Separated': 'single',
      'Widowed': 'single',
      'Married-spouse-absent':'married',
      'Married-AF-spouse': 'married'
  }
  df['marital.status'] = df['marital.status'].apply(lambda x: marital_status[x])
  df = pd.concat([df, pd.get_dummies(df['marital.status'],prefix='marital',prefix_sep=':')], axis=1)
  df.drop('marital.status', axis=1, inplace= True)

  df.drop('relationship', axis=1, inplace= True)
  df['occupation'] = np.where(df['occupation'] == '?','not recorded', df['occupation'])
  df = pd.concat([df, pd.get_dummies(df['occupation'],prefix='occupation',prefix_sep=':')], axis=1)
  df.drop('occupation', axis=1, inplace= True)

  df = pd.concat([df, pd.get_dummies(df['race'],prefix='race',prefix_sep=':')], axis=1)
  df.drop('race', axis=1, inplace= True)

  df['capital.gain'] = df['capital.gain'] - df['capital.loss']
  df.drop('capital.loss', axis=1, inplace= True)

  df['native.country'] = np.where((df['native.country'] == 'United-States' )|(df['native.country'] == 'Mexico'), df['native.country'], 'others')
  df = pd.concat([df, pd.get_dummies(df['native.country'],prefix='native.country',prefix_sep=':')], axis=1)
  df.drop('native.country', axis=1, inplace= True)

  df = pd.concat([df, pd.get_dummies(df['sex'],prefix='sex',prefix_sep=':')], axis=1)
  df.drop('sex', axis=1, inplace= True)

  
  # Normalize numerical feautures
  features_numerical = ['age', 'education', 'capital.gain', 'hours.per.week']
  mu = df[features_numerical].mean(axis=0)
  sigma  = df[features_numerical].std(axis=0)
  df[features_numerical] = (df[features_numerical] - mu) /sigma
  # add bias
  df['bias'] = 1
  # change the income column to "is greater than 50" and value of 1 for '>50K' and -1 for '<=50K'
  features = df.columns[(df.columns != 'income')]
  df['is greater than 50'] = np.where(df['income'] == '<=50K', -1, 1)
  df.drop('income', axis=1, inplace=True)
  # work on small portion of data for cheking quickly
  # df = df[:10]
  #shuffle
  df = df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
  #split
  indices = np.arange(len(df))
  training_filter = ((indices % 5) < 3)
  validation_filter = ((indices % 5) == 3)
  test_filter = ((indices % 5) == 4)
  train = df[training_filter]
  validation = df[validation_filter]
  test = df[test_filter]
  return train, validation, test

# Train your model on the dataset
def train():
  train, validation, test = prepare()
  train_x = train.drop('is greater than 50', axis=1)
  train_y =  np.array(train['is greater than 50'])
  validation_x = validation.drop('is greater than 50', axis=1)
  validation_y =  np.array(validation['is greater than 50'])
  test_x = test.drop('is greater than 50', axis=1)
  test_y = np.array(test['is greater than 50'])
  features = train_x.columns
  """ ### tuning hyper_parameter ###
  # best regularization term
  w0 = np.random.rand(train.shape[1]-1)
  model = LogisticRegression(w0, .2)
  cv_reg = np.array([0.2, 0.4, 0.8, 1.6])
  best_reg = model.tune_reg_term(train, validation_x, validation_y, cv_reg, feature_subset=features)"""
  # best_treshold
  # w0 = np.random.rand(train.shape[1]-1)
  # stepsize = .2
  # model = LogisticRegression(w0, .2)
  # best_treshold = model.tune_treshold(train, validation_x, validation_y, feature_subset=features)
  # train with chosen hyper_parameter
  w0 = np.random.rand(train.shape[1]-1)
  stepsize = .2
  best_reg = .2
  model = LogisticRegression(w0, best_reg)
  model.train(train, stepsize, 100, feature_subset=features)


def compute_accuracy(test_y, prediction):
    TP_filter = (test_y == +1) & (prediction == +1)
    TP = TP_filter.mean()
    TN_filter = (test_y == -1) & (prediction == -1)
    TN = TN_filter.mean()
    FP_filter = (test_y == -1) & (prediction == +1)
    FP = FP_filter.mean()
    FN_filter = (test_y == +1) & (prediction == -1)
    FN = FN_filter.mean()
    confusion_matrix = pd.DataFrame([[TP, FP],[TN, FN]], columns= ['True', 'False'], index=['>50K', '<=50K'])
    # print(confusion_matrix)
    return [TP, TN, FP, FN]

# Compute the evaluation metrics and figures
def evaluate():
  train, validation, test = prepare()
  test_x = test.drop('is greater than 50', axis=1)
  test_y = np.array(test['is greater than 50'])
  """ best w computed on train set and validated with validation set by 
    stepsize =  .2 and reg = .2
  """
  best_w = [0.116930, 0.171571, 0.192636, 0.122342, 0.020443,
          -0.001963, 0.000549, -0.030316, 0.014929, -0.025570, 
          -0.010216, 0.005816, -0.042982, 0.202440, -0.301690, 
          -0.013045, 0.001953, 0.011897, 0.065851, -0.027318, 
          -0.025603, -0.020434, -0.085476, -0.004454, 0.066130, 
          0.020224, 0.009021, 0.027985, -0.000234, -0.041186, 
          -0.004814, -0.004480, -0.045377, -0.000230, -0.031560, 
          -0.022982, -0.050920, -0.032904, -0.087182, -0.013076,
          -0.110634]
  model = LogisticRegression(best_w, .2)
  model.test_report(test_x, test_y)
  # mask = (test['is greater than 50'] == 1)
  # model.test_report(test_x[mask], test_y[mask])
  # compute_accuracy(test_y, model.predict(test_x))
  # Genarating figures for report
  # showing imbalanceness in the train set regarding gender
  lower_male = ((train['is greater than 50'] == -1) & (train['sex:Male'] == 1)).sum()
  lower_female = ((train['is greater than 50'] == -1) & (train['sex:Female'] == 1)).sum()
  greater_male = ((train['is greater than 50'] == +1) & (train['sex:Male'] == 1)).sum()
  greater_female = ((train['is greater than 50'] == +1) & (train['sex:Female'] == 1)).sum()

  lower = lower_male, lower_female 
  greater = greater_male, greater_female
  names = ['Male', 'Female']

  fig = plt.figure(figsize=(6,5), dpi=200)
  left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
  ax = fig.add_axes([left, bottom, width, height]) 
  
  width = 0.35   
  ticks = np.arange(len(names))    
  ax.bar(ticks, lower, width, label='<=50K')
  ax.bar(ticks + width, greater, width, align="center",
      label='>50K')

  ax.set_ylabel('Count')
  ax.set_title('Frequency of different genders in the train set')
  ax.set_xticks(ticks + width/2)
  ax.set_xticklabels(names)

  ax.legend(loc='best')
  plt.savefig('train_sex_detail.jpg')

  # showing imbalanceness in the train set regarding race
  lower_eskimo = ((train['is greater than 50'] == -1) & (train['race:Amer-Indian-Eskimo'] == 1)).sum()
  lower_asian_pac = ((train['is greater than 50'] == -1) & (train['race:Asian-Pac-Islander'] == 1)).sum()
  lower_black = ((train['is greater than 50'] == -1) & (train['race:Asian-Pac-Islander'] == 1)).sum()
  lower_other = ((train['is greater than 50'] == -1) & (train['race:Other'] == 1)).sum()
  lower_white = ((train['is greater than 50'] == -1) & (train['race:White'] == 1)).sum()
  greater_eskimo = ((train['is greater than 50'] == +1) & (train['race:Amer-Indian-Eskimo'] == 1)).sum()
  greater_asian_pac = ((train['is greater than 50'] == +1) & (train['race:Asian-Pac-Islander'] == 1)).sum()
  greater_black = ((train['is greater than 50'] == +1) & (train['race:Black'] == 1)).sum()
  greater_other = ((train['is greater than 50'] == +1) & (train['race:Other'] == 1)).sum()
  greater_white = ((train['is greater than 50'] == +1) & (train['race:White'] == 1)).sum()

  lower = lower_eskimo, lower_asian_pac, lower_black, lower_other, lower_white
  greater = greater_eskimo, greater_asian_pac, greater_black, greater_other, greater_white
  names = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']

  fig = plt.figure(figsize=(6,5), dpi=200)
  left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
  ax = fig.add_axes([left, bottom, width, height]) 
  
  width = 0.35   
  ticks = np.arange(len(names))    
  ax.bar(ticks, lower, width, label='<=50K')
  ax.bar(ticks + width, greater, width, align="center",
      label='>50K')

  ax.set_ylabel('Count')
  ax.set_title('Frequency of different races in the train set')
  ax.set_xticks(ticks + width/2)
  ax.set_xticklabels(names, rotation='vertical')

  ax.legend(loc='best')
  plt.savefig('train_race_detail.jpg')

  ### evaluate
  prediciton = model.predict(test_x)
  races = ['race:Amer-Indian-Eskimo','race:Asian-Pac-Islander', 'race:Black', 'race:Other', 'race:White']
  genders =  ['sex:Female', 'sex:Male']
  data = pd.DataFrame(columns=['group','TP', 'TN', 'FP', 'FN'])
  for gender in genders:
      mask = (test[gender] == 1)
      acc = compute_accuracy(test_y[mask], prediciton[mask])
      row = [gender, acc[0], acc[1], acc[2], acc[3]]
      data.loc[-1] = row
      data.index = data.index + 1  # shifting index
      data = data.sort_index()
  for race in races:
      mask = (test[race] == 1)
      acc = compute_accuracy(test_y[mask], prediciton[mask])
      row = [race, acc[0], acc[1], acc[2], acc[3]]
      data.loc[-1] = row
      data.index = data.index + 1  # shifting index
      data = data.sort_index()
  for gender in genders:
      for race in races:
          mask = (test[gender] == 1) & (test[race] == 1)
          acc = compute_accuracy(test_y[mask], prediciton[mask])
          subgroup = gender + ' ' + race
          row = [subgroup, acc[0], acc[1], acc[2], acc[3]]
          data.loc[-1] = row
          data.index = data.index + 1  # shifting index
          data = data.sort_index()
  prediciton = model.predict(test_x)
  acc = compute_accuracy(test_y, prediciton)
  row = ['All', acc[0], acc[1], acc[2], acc[3]]
  data.loc[-1] = row
  data.index = data.index + 1  # shifting index
  data = data.sort_index()
  data.plot(kind='barh',x='group',y='FP', fontsize=30, figsize=(65,30), title='False postive rate for threshold 0.5')
  plt.savefig('FP.jpg')
  data.plot(kind='barh',x='group',y='FN', fontsize=30, figsize=(65,30), title='False negative rate for threshold 0.5')
  plt.savefig('FN.jpg')
  data.plot(kind='barh',x='group',y='TP', fontsize=30, figsize=(65,30), title='True postive rate for threshold 0.5')
  plt.savefig('TP.jpg')
  data.plot(kind='barh',x='group',y='TN', fontsize=30, figsize=(65,30), title='True negative rate for threshold 0.5')
  plt.savefig('TN.jpg')


def build_paper():
  x = os.system('pdflatex paper.tex')
  x = os.system('pdflatex card.tex')


###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
  print("""
    You need to pass in a command-line argument.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
  """)
  sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
  supported_functions[arg]()
else:
  raise ValueError("""
    '{}' not among the allowed functions.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """.format(arg))
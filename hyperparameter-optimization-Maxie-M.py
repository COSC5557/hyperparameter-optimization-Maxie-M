##References: 
##https://www.projectpro.io/recipes/find-optimal-parameters-using-randomizedsearchcv-for-regression
##https://www.analyticsvidhya.com/blog/2022/02/a-comprehensive-guide-on-hyperparameter-tuning-and-its-techniques/

#importing needed tools
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, train_test_split

 #importing red wine data 
red_wine_data = pd.read_csv('winequality-red.csv')

#shape of red wine dataframe 
red_wine_data.shape

#checking the data type of the red wine dataset 
red_wine_data.dtypes

#information on red wine dataset 
red_wine_data.info()

#first 5 rows of red wine dataframe 
red_wine_data.head()

#all columns in red wine dataframe 
red_wine_data.columns

#untouched red wine data
red_wine_df = red_wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
red_wine_untouched = sns.boxplot(data = red_wine_df, orient = "h", palette = "Set2")
plt.show()

#binarization of the target variable, using list comprehension
red_wine_data['quality'] = [1 if x>=7 else 0 for x in red_wine_data['quality']]

#splitting into features and target 
X = red_wine_data.drop('quality', axis = 1)
y = red_wine_data['quality']

#splitting into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#defining a function to evaluate accuracy and handle scoring 
scorer = make_scorer(lambda y_true, y_pred: np.mean(y_true == y_pred))

#defining hyperparameter search space for each algorithm 
param_dist = {
    'decTr' : {
        'max_depth' : [None, 5, 10, 15], 
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4], 
        'max_features' : ['auto', 'sqrt', 'log2', None]
    },
    'ranFor' : {
        'n_estimators' : [50, 100, 150], 
        'max_depth' : [None, 5, 10, 15],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'max_features' : ['auto', 'sqrt', 'log2', None]
    }
}

#defining the models 
models = {
    'decTr' : DecisionTreeClassifier(),
    'ranFor' : RandomForestClassifier()
}

# Function to perform hyperparameter optimization
def optimize_hyperparameters(models, param_dist, X_train, y_train, scoring='accuracy', n_iter=10, cv=5, random_state=42):
    best_models = {}
    for model_name, model in models.items():
        print(f'Optimizing hyperparameters for {model_name}...')
        ran_search = RandomizedSearchCV(model, param_distributions=param_dist[model_name], scoring=scoring,
                                        n_iter=n_iter, cv=cv, random_state=random_state)
        ran_search.fit(X_train, y_train)
        best_models[model_name] = ran_search.best_estimator_
    return best_models

# Call the function to optimize hyperparameters
best_models = optimize_hyperparameters(models, param_dist, X_train, y_train)

#evaluating the best models 
for model_name, model in best_models.items(): 
    # Calculate AUC score 
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f'{model_name} AUC: {roc_auc:.4f}\n')

#function to plot ROC curves 
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for Different Models')
    plt.legend(loc='lower right')
    plt.show()

#call the function to plot ROC curves 
plot_roc_curves(best_models, X_test, y_test)

#performance comparison plot
def plot_performance_comparison(models, best_models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    model_names = list(models.keys())
    auc_scores_before = []
    auc_scores_after = []
    for model_name in model_names:
        #calculate AUC score before tuning
        y_pred_proba_before = models[model_name].fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_before)
        roc_auc_before = auc(fpr, tpr)
        auc_scores_before.append(roc_auc_before)
        
        #calculate AUC score after tuning
        y_pred_proba_after = best_models[model_name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_after)
        roc_auc_after = auc(fpr, tpr)
        auc_scores_after.append(roc_auc_after)
        
    #plot bars
    plt.bar(np.arange(len(model_names))-0.2, auc_scores_before, width=0.4, label='Before Tuning', color = 'lavender')
    plt.bar(np.arange(len(model_names))+0.2, auc_scores_after, width=0.4, label='After Tuning', color = 'mediumpurple')
    
    #add the percentages on top of bars
    for i in range(len(model_names)):
        plt.text(i - 0.2, auc_scores_before[i] + 0.01, f"{auc_scores_before[i]*100:.2f}%", ha='center')
        plt.text(i + 0.2, auc_scores_after[i] + 0.01, f"{auc_scores_after[i]*100:.2f}%", ha='center')
    
    plt.xticks(range(len(model_names)), model_names)
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.title('Performance Comparison Before and After Tuning')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#call the function to plot performance comparison
plot_performance_comparison(models, best_models, X_test, y_test)

#best configuration table 
def display_best_configuration(best_models):
    print('Best Configuration for Each Model:')
    print('-------------------------------------------')
    for model_name, model in best_models.items():
        print(f'{model_name}:')
        for param_name, param_value in model.get_params().items():
            print(f'{param_name}: {param_value}')
        print('---------------------------------------------')
        
#displaying the best configuration 
display_best_configuration(best_models)

#define the custom colormap ranging from lavender to purple
colors = ["lavender", "purple"]
cmap = mcolors.LinearSegmentedColormap.from_list("LavenderPurple", colors)

def visualize_best_configuration(best_models):
    config_dict = {}
    for model_name, model in best_models.items():
        model_config = model.get_params()
        config_dict[model_name] = model_config

    config_df = pd.DataFrame(config_dict).transpose()

    #labeling categorical hyperparameters
    label_encoder = LabelEncoder()
    for column in config_df.columns:
        if config_df[column].dtype == 'object':
            config_df[column] = label_encoder.fit_transform(config_df[column])

    plt.figure(figsize=(14, 8))  
    sns.set(font_scale=1.2) 
    sns.heatmap(config_df, annot=True, cmap=cmap, fmt=".2f", linewidths=.5)
    plt.title('Best Configuration for Each Model')
    plt.xlabel('Hyperparameter')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right') 
    plt.yticks(rotation=0) 
    plt.tight_layout() 
    plt.show()

#call the function to visualize best configuration
visualize_best_configuration(best_models)

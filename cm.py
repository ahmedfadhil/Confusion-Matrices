from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report

print("Accuracy: {.2f}".format(accuracy_score(y_test, tree_predicted)))
print("Precision: {.2f}".format(precision_score(y_test, tree_predicted)))
print("Recall: {.2f}".format(recall_score(y_test, tree_predicted)))
print("F1: {.2f}".format(f1_score(y_test, tree_predicted)))

print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))

print('Random class proportional (dummy)\n',
      classification_report(y_test, y_classprop_predicted, target_names=['not 1', '1']))
print('SVM\n',
      classification_report(y_test, svm_predicted, target_names=['not 1', '1']))
print('Logistic Regression\n',
      classification_report(y_test, lr_predicted, target_names=['not 1', '1']))
print('Decision Tree\n',
      classification_report(y_test, tree_predicted, target_names=['not 1', '1']))

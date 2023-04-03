from sklearn import svm

###############################################################################
#                                   training                                  #
###############################################################################

classifier = svm.LinearSVC(
    penalty='l2', # default
    loss='squared_hinge', # default
    dual=True, # default
    tol=0.0001, # default
    C=1.0, # default
    fit_intercept=True, # default
    intercept_scaling=1.0, # default
    class_weight='balanced', # NOT default
    random_state=855, # NOT default
    max_iter=10000, # NOT default, changed from default(1000), 011122 Gabby
)

def trained_classifier(X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier

def train(X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier.decision_function

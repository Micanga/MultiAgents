
from sklearn.ensemble import RandomForestClassifier

def trained_models():
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=12345)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    svc_w_linear_kernel = SVC(kernel='linear')
    svc_w_linear_kernel.fit(X_train, y_train)

    svc_wo_linear_kernel = SVC()
    svc_wo_linear_kernel.fit(X_train, y_train)

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    return {'RF':rf, 'LR':lr, 'SVC_w_linear_kernel':svc_w_linear_kernel,
            'Dummy':dummy, 'SVC_wo_linear_kernel':svc_wo_linear_kernel}




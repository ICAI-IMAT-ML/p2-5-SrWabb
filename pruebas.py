from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.Lab2_5_LogisticRegression_and_regularization import LogisticRegressor
import numpy as np

def sample_data():
    """Create a simple dataset for testing"""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_fit_basic(sample_data):
    """Test basic fitting without regularization"""
    model = LogisticRegressor()
    X_train, _, y_train, _ = sample_data
    print(X_train.shape)
    print("CACA")
    print(y_train)
    model.fit(X_train, y_train, num_iterations=100)

#test_fit_basic(sample_data())
a = np.array([[1,3],[2,6],[3,9]])
b = [1,2,3]
print(a.shape)
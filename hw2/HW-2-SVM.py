#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
import random
from collections import Counter
import time
import copy
import matplotlib.pyplot as plt

def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)


def load_and_shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "/Users/tapankhaladkar/Machine Learning/hw2/data/pos"
    neg_path = "/Users/tapankhaladkar/Machine Learning/hw2/data/neg"

    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    random.shuffle(review)
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale
        
        
def bow(words):
    return dict(Counter(words))



def pegasos_sparse(X, y, lambda_param, max_epochs=1000, tolerance=0.001):
    n = len(X)
    w = {}  
    t = 0  
    
    for epoch in range(max_epochs):
        misclassified = 0
        w_previous = copy.deepcopy(w) 
        
        data = list(zip(X, y))
        random.shuffle(data)
        
        for x_i, y_i in data:
            t += 1
            eta = 1 / (lambda_param * t)
            
            if dotProduct(w, x_i) * y_i < 1:
                increment(w, -lambda_param * eta, w)  # w = (1 - eta * lambda) * w
                increment(w, eta * y_i, x_i)  # w += eta * y_i * x_i
                misclassified += 1
            else:
                increment(w, -lambda_param * eta, w)  # w = (1 - eta * lambda) * w
        
        error_rate = misclassified / n
        if error_rate < tolerance:
            print(f"Converged after {epoch + 1} epochs (Error rate: {error_rate:.6f})")
            break
        
        weight_change = sum((w.get(k, 0) - w_previous.get(k, 0))**2 for k in set(w) | set(w_previous))
        if weight_change < tolerance:
            print(f"Converged after {epoch + 1} epochs (Weight change: {weight_change:.6f})")
            break
    
    return w

def pegasos_sparse_optimized(X, y, lambda_param, max_epochs=1000, tolerance=0.001):
    n = len(X)
    s = 1  
    W = {}  
    t = 2  

    for epoch in range(max_epochs):
        misclassified = 0
        
        data = list(zip(X, y))
        random.shuffle(data)
        
        for x_i, y_i in data:
            eta = 1 / (lambda_param * t)
            
            s = s * (1 - eta * lambda_param)
            
            if s == 0:
                s = 1
                W.clear()  
            else:
                if y_i * dotProduct(W, x_i) < 1/s:
                    increment(W, (eta * y_i) / s, x_i)
                    misclassified += 1
            
            t += 1
        
        error_rate = misclassified / n
        if error_rate < tolerance:
            print(f"Converged after {epoch + 1} epochs (Error rate: {error_rate:.6f})")
            break
    
    return s, W


def compute_norm(w):
    return sum(v**2 for v in w.values())**0.5

def predict_original(w, x):
    return 1 if dotProduct(w, x) >= 0 else -1

def predict_optimized(s, W, x):
    return 1 if dotProduct(W, x) >= 0 else -1


def classification_error(w, X, y):
    errors = sum(1 for x_i, y_i in zip(X, y) if y_i * dotProduct(w, x_i) < 0)
    return errors / len(y)

def search_lambda(X_train, y_train, X_val, y_val, lambda_range):
    results = []
    for lambda_param in lambda_range:
        s, W = pegasos_sparse_optimized(X_train, y_train, lambda_param)
        val_error = classification_error({k: v*s for k, v in W.items()}, X_val, y_val)
        results.append((lambda_param, val_error))
        print(f"λ = {lambda_param:.6f}, Validation Error = {val_error:.4f}")
    return results


all_data = load_and_shuffle_data()
train_data = all_data[:1500]
val_data = all_data[1500:]

X_train = []
y_train = []
for review in train_data:
    X_train.append(bag_of_words(review[:-1]))  # Convert review text to bag-of-words
    y_train.append(review[-1])  


X_val = []
y_val = []
for review in val_data:
    X_val.append(bag_of_words(review[:-1]))  
    y_val.append(review[-1])  

# Print some statistics
print(f"Number of training examples: {len(X_train)}")
print(f"Number of validation examples: {len(X_val)}")
print(f"Sample training example:\n{list(X_train[0].items())[:5]}...")
print(f"Corresponding label: {y_train[0]}")

max_epochs = 2
lambda_param = 0.1

start_time = time.time()
w = pegasos_sparse(X_train, y_train, lambda_param, max_epochs)
original_time = time.time() - start_time

start_time = time.time()
s, W = pegasos_sparse_optimized(X_train, y_train, lambda_param, max_epochs)
optimized_time = time.time() - start_time

norm_original = compute_norm(w)
norm_optimized = s * compute_norm(W)

print(f"Original Pegasos runtime: {original_time:.4f} seconds")
print(f"Optimized Pegasos runtime: {optimized_time:.4f} seconds")
print(f"Norm of original weight vector: {norm_original:.6f}")
print(f"Norm of optimized weight vector: {norm_optimized:.6f}")
print(f"Relative difference in norms: {abs(norm_original - norm_optimized) / norm_original:.6f}")

num_checks = 100
mismatches = 0
for x in X_train[:num_checks]:
    pred_original = predict_original(w, x)
    pred_optimized = predict_optimized(s, W, x)
    if pred_original != pred_optimized:
        mismatches += 1

print(f"Prediction mismatches in {num_checks} samples: {mismatches}")

top_features_original = sorted(w.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
top_features_optimized = sorted(((k, v*s) for k, v in W.items()), key=lambda x: abs(x[1]), reverse=True)[:10]

print("\nTop 10 features (original):")
for feature, weight in top_features_original:
    print(f"{feature}: {weight:.6f}")

print("\nTop 10 features (optimized):")
for feature, weight in top_features_optimized:
    print(f"{feature}: {weight:.6f}")
    
error = classification_error(w, X_train, y_train)
#print(f"Classification error: {error:.4f}")
lambda_range = np.logspace(-6, 0, num=7)
results = search_lambda(X_train, y_train, X_val, y_val, lambda_range)

lambda_values, val_errors = zip(*results)
plt.figure(figsize=(10, 6))
plt.semilogx(lambda_values, val_errors, 'bo-')
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('Validation Error')
plt.title('Validation Error vs Regularization Parameter')
plt.grid(True)
plt.show()

# Find the best lambda from this search
best_lambda, best_error = min(results, key=lambda x: x[1])
print(f"\nBest λ from initial search: {best_lambda:.6f}")
print(f"Best validation error: {best_error:.4f}")

# Refined search around the best lambda
refined_lambda_range = np.logspace(np.log10(best_lambda/10), np.log10(best_lambda*10), num=10)
refined_results = search_lambda(X_train, y_train, X_val, y_val, refined_lambda_range)

# Plot refined results
refined_lambda_values, refined_val_errors = zip(*refined_results)
plt.figure(figsize=(10, 6))
plt.semilogx(refined_lambda_values, refined_val_errors, 'ro-')
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('Validation Error')
plt.title('Refined Search: Validation Error vs Regularization Parameter')
plt.grid(True)
plt.show()

best_lambda_refined, best_error_refined = min(refined_results, key=lambda x: x[1])
print(f"\nBest λ from refined search: {best_lambda_refined:.6f}")
print(f"Best validation error: {best_error_refined:.4f}")

# Train the final model with the best lambda
s_final, W_final = pegasos_sparse_optimized(X_train, y_train, best_lambda_refined)

# Compute the test error
X_test = [bow(review[:-1]) for review in val_data]  # Using validation data as test data
y_test = [review[-1] for review in val_data]
test_error = classification_error({k: v*s_final for k, v in W_final.items()}, X_test, y_test)
print(f"\nFinal test error with best λ: {test_error:.4f}")


# In[ ]:


import os
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt
import copy

def read_data(file):
    with open(file, 'r') as f:
        lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = [word.translate(str.maketrans("", "", symbols)).strip() for word in lines]
    return list(filter(None, words))

def folder_list(path, label):
    return [read_data(os.path.join(path, file)) + [label] for file in os.listdir(path)]

def load_and_shuffle_data():
    pos_path = "/Users/tapankhaladkar/Machine Learning/hw2/data/pos"
    neg_path = "/Users/tapankhaladkar/Machine Learning/hw2/data/neg"
    data = folder_list(pos_path, 1) + folder_list(neg_path, -1)
    random.shuffle(data)
    return data

def bow(words):
    return dict(Counter(words))

# Helper functions for sparse vectors
def dotProduct(d1, d2):
    return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale
        
def pegasos_sparse(X, y, lambda_param, max_epochs=1000, tolerance=0.001):
    n = len(X)
    w = {}  
    t = 0  
    
    for epoch in range(max_epochs):
        misclassified = 0
        w_previous = copy.deepcopy(w) 
        
        data = list(zip(X, y))
        random.shuffle(data)
        
        for x_i, y_i in data:
            t += 1
            eta = 1 / (lambda_param * t)
            
            if dotProduct(w, x_i) * y_i < 1:
                increment(w, -lambda_param * eta, w)  # w = (1 - eta * lambda) * w
                increment(w, eta * y_i, x_i)  # w += eta * y_i * x_i
                misclassified += 1
            else:
                increment(w, -lambda_param * eta, w)  # w = (1 - eta * lambda) * w
        
        error_rate = misclassified / n
        if error_rate < tolerance:
            print(f"Converged after {epoch + 1} epochs (Error rate: {error_rate:.6f})")
            break
        
        weight_change = sum((w.get(k, 0) - w_previous.get(k, 0))**2 for k in set(w) | set(w_previous))
        if weight_change < tolerance:
            print(f"Converged after {epoch + 1} epochs (Weight change: {weight_change:.6f})")
            break
    
    return w


# Pegasos algorithm implementations
def pegasos_sparse_optimized(X, y, lambda_param, max_epochs=1000, tolerance=0.001):
    n = len(X)
    s = 1  
    W = {}  
    t = 2  

    for epoch in range(max_epochs):
        misclassified = 0
        
        data = list(zip(X, y))
        random.shuffle(data)
        
        for x_i, y_i in data:
            eta = 1 / (lambda_param * t)
            
            s = s * (1 - eta * lambda_param)
            
            if s == 0:
                s = 1
                W.clear()  
            else:
                if y_i * dotProduct(W, x_i) < 1/s:
                    increment(W, (eta * y_i) / s, x_i)
                    misclassified += 1
            
            t += 1
        
        error_rate = misclassified / n
        if error_rate < tolerance:
            print(f"Converged after {epoch + 1} epochs (Error rate: {error_rate:.6f})")
            break
    
    return s, W

def compute_norm(w):
    return sum(v**2 for v in w.values())**0.5

def predict_original(w, x):
    return 1 if dotProduct(w, x) >= 0 else -1

def predict_optimized(s, W, x):
    return 1 if dotProduct(W, x) >= 0 else -1


# Evaluation functions
def classification_error(w, X, y):
    return sum(1 for x_i, y_i in zip(X, y) if y_i * dotProduct(w, x_i) < 0) / len(y)


def search_optimal_lambda(X_train, y_train, X_val, y_val, max_epochs=1000, tolerance=0.001):
    lambdas = np.logspace(-5, 1, num=10)  # From 1e-5 to 10
    errors = []

    for lambda_param in lambdas:
        print(f"Testing lambda: {lambda_param:.5f}")
        s, W = pegasos_sparse_optimized(X_train, y_train, lambda_param, max_epochs, tolerance)
        error = classification_error({k: v * s for k, v in W.items()}, X_val, y_val)
        print(f"Validation error: {error:.4f}")
        errors.append(error)

    return lambdas, errors


def plot_errors(lambdas, errors):
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, errors, marker='o', linestyle='-', color='b')
    plt.xscale('log')  # Log scale for lambda
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('Classification Error')
    plt.title('Classification Error vs Regularization Parameter (λ)')
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    all_data = load_and_shuffle_data()
    train_data = all_data[:1500]
    val_data = all_data[1500:]
    
    X_train = [bow(review[:-1]) for review in train_data]
    y_train = [review[-1] for review in train_data]
    X_val = [bow(review[:-1]) for review in val_data]
    y_val = [review[-1] for review in val_data]

    print(f"Number of training examples: {len(X_train)}")
    print(f"Number of validation examples: {len(X_val)}")
    print(f"Sample training example:\n{list(X_train[0].items())[:5]}...")
    print(f"Corresponding label: {y_train[0]}")
    
    max_epochs = 2
    lambda_param = 0.1

    start_time = time.time()
    w = pegasos_sparse(X_train, y_train, lambda_param, max_epochs)
    original_time = time.time() - start_time

    start_time = time.time()
    s, W = pegasos_sparse_optimized(X_train, y_train, lambda_param, max_epochs)
    optimized_time = time.time() - start_time

    norm_original = compute_norm(w)
    norm_optimized = s * compute_norm(W)

    print(f"Original Pegasos runtime: {original_time:.4f} seconds")
    print(f"Optimized Pegasos runtime: {optimized_time:.4f} seconds")
    print(f"Norm of original weight vector: {norm_original:.6f}")
    print(f"Norm of optimized weight vector: {norm_optimized:.6f}")
    print(f"Relative difference in norms: {abs(norm_original - norm_optimized) / norm_original:.6f}")
    
    error_original = classification_error(w, X_val, y_val)
    error_optimized = classification_error({k: v*s for k, v in W.items()}, X_val, y_val)
    print(f"Original Pegasos classification error: {error_original:.4f}")
    print(f"Optimized Pegasos classification error: {error_optimized:.4f}")

    
    lambdas, errors = search_optimal_lambda(X_train, y_train, X_val, y_val)
    plot_errors(lambdas, errors)

    # Find the best lambda and its corresponding error
    best_lambda = lambdas[np.argmin(errors)]
    best_error = min(errors)

    print(f"Best regularization parameter (λ): {best_lambda:.5f}")
    print(f"Minimum classification error: {best_error:.4f}")
    


# In[ ]:


import os
import numpy as np
import random
from collections import Counter
import time
import matplotlib.pyplot as plt
import copy

def read_data(file):
    with open(file, 'r') as f:
        lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = [word.translate(str.maketrans("", "", symbols)).strip() for word in lines]
    return list(filter(None, words))

def folder_list(path, label):
    return [read_data(os.path.join(path, file)) + [label] for file in os.listdir(path)]

def load_and_shuffle_data():
    pos_path = "/Users/tapankhaladkar/Machine Learning/hw2/data/pos"
    neg_path = "/Users/tapankhaladkar/Machine Learning/hw2/data/neg"
    data = folder_list(pos_path, 1) + folder_list(neg_path, -1)
    random.shuffle(data)
    return data

def bow(words):
    return dict(Counter(words))

# Helper functions for sparse vectors
def dotProduct(d1, d2):
    return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale
        
def pegasos_sparse(X, y, lambda_param, max_epochs=1000, tolerance=0.001):
    n = len(X)
    w = {}  
    t = 0  
    
    for epoch in range(max_epochs):
        misclassified = 0
        w_previous = copy.deepcopy(w) 
        
        data = list(zip(X, y))
        random.shuffle(data)
        
        for x_i, y_i in data:
            t += 1
            eta = 1 / (lambda_param * t)
            
            if dotProduct(w, x_i) * y_i < 1:
                increment(w, -lambda_param * eta, w)  # w = (1 - eta * lambda) * w
                increment(w, eta * y_i, x_i)  # w += eta * y_i * x_i
                misclassified += 1
            else:
                increment(w, -lambda_param * eta, w)  # w = (1 - eta * lambda) * w
        
        error_rate = misclassified / n
        if error_rate < tolerance:
            print(f"Converged after {epoch + 1} epochs (Error rate: {error_rate:.6f})")
            break
        
        weight_change = sum((w.get(k, 0) - w_previous.get(k, 0))**2 for k in set(w) | set(w_previous))
        if weight_change < tolerance:
            print(f"Converged after {epoch + 1} epochs (Weight change: {weight_change:.6f})")
            break
    
    return w


# Pegasos algorithm implementations
def pegasos_sparse_optimized(X, y, lambda_param, max_epochs=1000, tolerance=0.001):
    n = len(X)
    s = 1  
    W = {}  
    t = 2  

    for epoch in range(max_epochs):
        misclassified = 0
        
        data = list(zip(X, y))
        random.shuffle(data)
        
        for x_i, y_i in data:
            eta = 1 / (lambda_param * t)
            
            s = s * (1 - eta * lambda_param)
            
            if s == 0:
                s = 1
                W.clear()  
            else:
                if y_i * dotProduct(W, x_i) < 1/s:
                    increment(W, (eta * y_i) / s, x_i)
                    misclassified += 1
            
            t += 1
        
        error_rate = misclassified / n
        if error_rate < tolerance:
            print(f"Converged after {epoch + 1} epochs (Error rate: {error_rate:.6f})")
            break
    
    return s, W

def compute_norm(w):
    return sum(v**2 for v in w.values())**0.5

def predict_original(w, x):
    return 1 if dotProduct(w, x) >= 0 else -1

def predict_optimized(s, W, x):
    return 1 if dotProduct(W, x) >= 0 else -1


# Evaluation functions
def classification_error(w, X, y):
    return sum(1 for x_i, y_i in zip(X, y) if y_i * dotProduct(w, x_i) < 0) / len(y)


def search_optimal_lambda(X_train, y_train, X_val, y_val, max_epochs=1000, tolerance=0.001):
    lambdas = np.logspace(-5, 1, num=10)  # From 1e-5 to 10
    errors = []

    for lambda_param in lambdas:
        print(f"Testing lambda: {lambda_param:.5f}")
        s, W = pegasos_sparse_optimized(X_train, y_train, lambda_param, max_epochs, tolerance)
        error = classification_error({k: v * s for k, v in W.items()}, X_val, y_val)
        print(f"Validation error: {error:.4f}")
        errors.append(error)

    return lambdas, errors


def plot_errors(lambdas, errors):
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, errors, marker='o', linestyle='-', color='b')
    plt.xscale('log')  # Log scale for lambda
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('Classification Error')
    plt.title('Classification Error vs Regularization Parameter (λ)')
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    all_data = load_and_shuffle_data()
    train_data = all_data[:1500]
    val_data = all_data[1500:]
    
    X_train = [bow(review[:-1]) for review in train_data]
    y_train = [review[-1] for review in train_data]
    X_val = [bow(review[:-1]) for review in val_data]
    y_val = [review[-1] for review in val_data]

    print(f"Number of training examples: {len(X_train)}")
    print(f"Number of validation examples: {len(X_val)}")
    print(f"Sample training example:\n{list(X_train[0].items())[:5]}...")
    print(f"Corresponding label: {y_train[0]}")
    
    max_epochs = 2
    lambda_param = 0.1

    start_time = time.time()
    w = pegasos_sparse(X_train, y_train, lambda_param, max_epochs)
    original_time = time.time() - start_time

    start_time = time.time()
    s, W = pegasos_sparse_optimized(X_train, y_train, lambda_param, max_epochs)
    optimized_time = time.time() - start_time

    norm_original = compute_norm(w)
    norm_optimized = s * compute_norm(W)

    print(f"Original Pegasos runtime: {original_time:.4f} seconds")
    print(f"Optimized Pegasos runtime: {optimized_time:.4f} seconds")
    print(f"Norm of original weight vector: {norm_original:.6f}")
    print(f"Norm of optimized weight vector: {norm_optimized:.6f}")
    print(f"Relative difference in norms: {abs(norm_original - norm_optimized) / norm_original:.6f}")
    
    error_original = classification_error(w, X_val, y_val)
    error_optimized = classification_error({k: v*s for k, v in W.items()}, X_val, y_val)
    print(f"Original Pegasos classification error: {error_original:.4f}")
    print(f"Optimized Pegasos classification error: {error_optimized:.4f}")

    
    lambdas, errors = search_optimal_lambda(X_train, y_train, X_val, y_val)
    plot_errors(lambdas, errors)

    best_lambda = lambdas[np.argmin(errors)]
    best_error = min(errors)

    print(f"Best regularization parameter (λ): {best_lambda:.5f}")
    print(f"Minimum classification error: {best_error:.4f}")
    


# In[ ]:





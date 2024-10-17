Practical 1
Cmd installation:
pip install folium
pip install networkx
pip install matplotlib
Code:
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.lines import Line2D

# Graph represented as an adjacency list (city connections)
graph = {
    'Mumbai': ['Pune', 'Delhi', 'Bangalore'],
    'Pune': ['Mumbai', 'Nagpur'],
    'Delhi': ['Mumbai', 'Chandigarh'],
    'Bangalore': ['Mumbai', 'Hyderabad'],
    'Nagpur': ['Pune'],
    'Chandigarh': ['Delhi'],
    'Hyderabad': ['Bangalore']
}

# Breadth First Search (BFS) with limit
def bfs(graph, start_node, limit=None):
    visited = set()
    queue = deque([start_node])
    
    bfs_traversal = []
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            bfs_traversal.append(node)
            visited.add(node)
            queue.extend(graph[node])
        
        if limit and len(bfs_traversal) >= limit:
            break
    
    return bfs_traversal

# Iterative Depth First Search (IDFS) with limit
def idfs(graph, start_node, limit=None):
    visited = set()
    stack = [(start_node, 0)]
    idfs_traversal = []
    
    while stack:
        node, depth = stack.pop()
        if node not in visited:
            idfs_traversal.append(node)
            visited.add(node)
            stack.extend((neighbor, depth + 1) for neighbor in graph[node])
        
        if limit and len(idfs_traversal) >= limit:
            break
    
    return idfs_traversal

# Plotting the graph and traversal
def plot_graph(graph, traversal=None, title="City Connections"):
    G = nx.Graph()
    
    # Add edges between cities
    for city, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(city, neighbor)
    
    pos = nx.spring_layout(G)  # Layout for the nodes
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='black', font_size=10, font_color='black', font_weight='bold')
    
    # Highlight the traversal path if provided
    if traversal:
        edges_in_path = [(traversal[i], traversal[i+1]) for i in range(len(traversal)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, width=4, edge_color='orange')
        nx.draw_networkx_nodes(G, pos, nodelist=traversal, node_color='orange', node_size=2000)
    
    plt.title(title)
    plt.show()

# Plot both BFS and IDFS in one graph
def plot_comparison(graph, bfs_traversal, idfs_traversal):
    G = nx.Graph()
    
    for city, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(city, neighbor)
    
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(8, 6))
    
    # Plot BFS edges in orange
    edges_in_bfs = [(bfs_traversal[i], bfs_traversal[i+1]) for i in range(len(bfs_traversal)-1)]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='black', font_size=10, font_color='black', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_bfs, width=4, edge_color='orange')
    nx.draw_networkx_nodes(G, pos, nodelist=bfs_traversal, node_color='orange', node_size=2000)

    # Plot IDFS edges in green
    edges_in_idfs = [(idfs_traversal[i], idfs_traversal[i+1]) for i in range(len(idfs_traversal)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_idfs, width=4, edge_color='green')
    nx.draw_networkx_nodes(G, pos, nodelist=idfs_traversal, node_color='green', node_size=2000)
    
    # Create custom legend with colors
    legend_elements = [Line2D([0], [0], color='orange', lw=4, label='BFS Path'),
                       Line2D([0], [0], color='green', lw=4, label='IDFS Path')]
    
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title("Comparison of BFS and IDFS Traversals (Limited to 4 cities)")
    plt.show()

# Step 1: BFS Traversal (with limit)
bfs_result = bfs(graph, 'Mumbai', limit=4)
print("BFS Traversal: ", bfs_result)

# Step 2: IDFS Traversal (with limit)
idfs_result = idfs(graph, 'Mumbai', limit=4)
print("IDFS Traversal: ", idfs_result)

# Step 3: Plot the graph for BFS 
plot_graph(graph, bfs_result, "BFS Traversal (Limited to 4 cities)")

# Step 4: Plot the graph for IDFS 
plot_graph(graph, idfs_result, "IDFS Traversal (Limited to 4 cities)")

# Step 6: Plot comparison between BFS and IDFS traversals
plot_comparison(graph, bfs_result, idfs_result)
 
Practical 2
Cmd installation:
pip install folium
pip install networkx
pip install matplotlib
Code:
import folium
import networkx as nx
import heapq
import time
import matplotlib.pyplot as plt

# Define city connections in adjacency list format
city_connections = {
    'Mumbai': ['Pune', 'Nashik', 'Goa'],
    'Pune': ['Mumbai', 'Aurangabad', 'Satara'],
    'Nagpur': ['Aurangabad'],
    'Nashik': ['Mumbai', 'Aurangabad'],
    'Aurangabad': ['Pune', 'Nagpur', 'Nashik'],
    'Goa': ['Mumbai', 'Kolhapur'],
    'Satara': ['Pune', 'Kolhapur'],
    'Kolhapur': ['Satara', 'Goa'],
}

# Create the graph
G = nx.Graph()
for city, connections in city_connections.items():
    for neighbor in connections:
        G.add_edge(city, neighbor)

# A* Search function
def a_star_search(graph, start, goal):
    queue = []
    heapq.heappush(queue, (0, start, [start]))
    visited = set()

    while queue:
        _, current, path = heapq.heappop(queue)
        if current == goal:
            return path
        
        visited.add(current)
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                heapq.heappush(queue, (0, neighbor, path + [neighbor]))
    
    return None

# Recursive Best-First Search function
def rbfs(graph, current, goal, f_limit, path):
    if current == goal:
        return path

    successors = []
    for neighbor in graph.neighbors(current):
        if neighbor not in path:
            successors.append(neighbor)

    if not successors:
        return None

    while successors:
        best = successors[0]
        result = rbfs(graph, best, goal, f_limit, path + [best])
        if result:
            return result
        
        successors.pop(0)

    return None

# Helper function for RBFS
def recursive_best_first_search(graph, start, goal):
    return rbfs(graph, start, goal, float('inf'), [start])

# Execute A* Search
start = 'Mumbai'
goal = 'Kolhapur'

start_time = time.time()
a_star_result = a_star_search(G, start, goal)
a_star_time = time.time() - start_time

print("Path from Mumbai to Kolhapur using A* Search:", a_star_result)
print("A* Search Time:", a_star_time, "seconds")

# Execute RBFS
start_time = time.time()
rbfs_result = recursive_best_first_search(G, start, goal)
rbfs_time = time.time() - start_time

print("Path from Mumbai to Kolhapur using RBFS:", rbfs_result)
print("RBFS Time:", rbfs_time, "seconds")

# Create the map with Folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=6)

# Add city markers
for city in city_connections.keys():
    folium.Marker(
        location=[19.0760, 72.8777],  # You can set a default location for markers
        popup=city,
    ).add_to(m)

# Add paths to the map
for path in [a_star_result, rbfs_result]:
    if path:
        for i in range(len(path) - 1):
            folium.PolyLine(
                locations=[[19.0760, 72.8777], [19.0760, 72.8777]],  # Dummy coordinates for illustration
                color='blue' if path == a_star_result else 'green',
                weight=2.5,
                opacity=0.7
            ).add_to(m)

# Function to plot a path in the graph
def plot_path(graph, path, title, color):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')
    
    # Highlight the path
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=color, width=3)

    plt.title(title)
    plt.show()

# Plotting graphs
plot_path(G, a_star_result, "A* Search Path from Mumbai to Kolhapur", 'blue')
plot_path(G, rbfs_result, "RBFS Path from Mumbai to Kolhapur", 'green')

# Comparison graph for both A* Search and RBFS
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')

# Highlight A* Search path
if a_star_result:
    a_star_edges = list(zip(a_star_result[:-1], a_star_result[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=a_star_edges, edge_color='blue', width=3, label='A* Path')

# Highlight RBFS path
if rbfs_result:
    rbfs_edges = list(zip(rbfs_result[:-1], rbfs_result[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=rbfs_edges, edge_color='green', width=3, label='RBFS Path')

plt.title("Comparison of A* Search and RBFS Paths")
plt.legend()
plt.show()
 
Practical 3
Cmd installation:
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install numpy pandas matplotlib seaborn scikit-learn
Code:
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("First 5 rows of the dataset:")
print(df.head())
print("\nShape of the dataset:", df.shape)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Build the Original Decision Tree Classifier
original_decision_tree = DecisionTreeClassifier(random_state=42)
original_decision_tree.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred_original = original_decision_tree.predict(X_test)

# Step 5: Evaluate the original model
accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"\nAccuracy of Original Decision Tree Classifier: {accuracy_original * 100:.2f}%")

# Confusion Matrix for the original model
conf_matrix_original = confusion_matrix(y_test, y_pred_original)
print("\nConfusion Matrix (Original):")
print(conf_matrix_original)

# Classification Report for the original model
print("\nClassification Report (Original):")
print(classification_report(y_test, y_pred_original, target_names=iris.target_names))

# Step 6: Build the Pruned Decision Tree Classifier
max_depth = 3  # Set the maximum depth for pruning
pruned_decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
pruned_decision_tree.fit(X_train, y_train)

# Step 7: Make predictions on the test set for the pruned tree
y_pred_pruned = pruned_decision_tree.predict(X_test)

# Step 8: Evaluate the pruned model
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"\nAccuracy of Pruned Decision Tree Classifier: {accuracy_pruned * 100:.2f}%")

# Confusion Matrix for the pruned model
conf_matrix_pruned = confusion_matrix(y_test, y_pred_pruned)
print("\nConfusion Matrix (Pruned):")
print(conf_matrix_pruned)

# Classification Report for the pruned model
print("\nClassification Report (Pruned):")
print(classification_report(y_test, y_pred_pruned, target_names=iris.target_names))

# Step 9: Visualize the Original Decision Tree
plt.figure(figsize=(12, 8))
plt.title("Original Decision Tree Visual Representation", fontsize=16)  # Ensure title is clear
plot_tree(original_decision_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.suptitle("Original Decision Tree", fontsize=20)  # Add a main title
plt.tight_layout()  # Adjust layout
plt.show()

# Step 10: Visualize the Pruned Decision Tree
plt.figure(figsize=(12, 8))
plt.title("Pruned Decision Tree Visual Representation", fontsize=16)  # Ensure title is clear
plot_tree(pruned_decision_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.suptitle("Pruned Decision Tree", fontsize=20)  # Add a main title
plt.tight_layout()  # Adjust layout
plt.show() 
Practical 4
Cmd installation:
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
Code:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, confusion_matrix 
 
# Step 1: Load and Prepare the Data 
wine = load_wine() 
 
X = wine.data 
y = wine.target 
 
# Convert to DataFrame for visualization 
data = pd.DataFrame(X, columns=wine.feature_names) 
data['target'] = y 
 
print(data.describe()) 
print() 
print("Shape of the dataset is:", data.shape) 
print() 
 
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
 
# Normalize the data 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
 
# Step 2: Build and Train the Feed Forward Neural Network 
 
# Activation function and its derivative 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def sigmoid_derivative(x): 
    return x * (1 - x) 
 
# Neural Network parameters 
input_layer_size = X_train.shape[1] 
hidden_layer_size = 10  # Number of neurons in hidden layer 
output_layer_size = 3    # Three classes in the wine dataset 
learning_rate = 0.01 
epochs = 10000 
 
# Initialize weights 
np.random.seed(42) 
weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size) 
weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size) 
 
# Training the Neural Network 
mse_values = [] 
for epoch in range(epochs): 
    # Feedforward 
    hidden_layer_input = np.dot(X_train, weights_input_hidden) 
    hidden_layer_output = sigmoid(hidden_layer_input) 
     
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) 
    final_output = sigmoid(final_layer_input) 
     
    # One-hot encode the output 
    y_train_onehot = np.zeros((y_train.size, y_train.max() + 1)) 
    y_train_onehot[np.arange(y_train.size), y_train] = 1 
     
    # Compute the error 
    error = y_train_onehot - final_output 
     
    # Backpropagation 
    d_final_output = error * sigmoid_derivative(final_output) 
    error_hidden_layer = np.dot(d_final_output, weights_hidden_output.T) 
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output) 
     
    # Update weights 
    weights_hidden_output += np.dot(hidden_layer_output.T, d_final_output) * learning_rate 
    weights_input_hidden += np.dot(X_train.T, d_hidden_layer) * learning_rate 
     
    mse = np.mean(error**2) 
    mse_values.append(mse) 
    if epoch % 1000 == 0: 
        print(f'Epoch {epoch}/{epochs}, MSE: {mse}') 
 
# Visualize the training process 
plt.figure(figsize=(10, 6)) 
plt.plot(mse_values, label="MSE during Training") 
plt.xlabel("Epochs") 
plt.ylabel("Mean Squared Error") 
plt.title("Training Progress of the Neural Network") 
plt.legend() 
plt.show() 
 
# Step 3: Evaluate the Neural Network 
hidden_layer_input_test = np.dot(X_test, weights_input_hidden) 
hidden_layer_output_test = sigmoid(hidden_layer_input_test) 
 
final_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) 
final_output_test = sigmoid(final_layer_input_test) 
 
# Convert final output to class predictions 
predictions = np.argmax(final_output_test, axis=1) 
 
# Evaluate performance 
accuracy = accuracy_score(y_test, predictions) 
print(f'Accuracy: {accuracy * 100:.2f}%') 
 
# Confusion Matrix 
conf_matrix = confusion_matrix(y_test, predictions) 
plt.figure(figsize=(8, 6)) 
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, 
yticklabels=wine.target_names) 
plt.xlabel("Predicted Labels") 
plt.ylabel("True Labels") 
plt.title("Confusion Matrix of the Neural Network Predictions") 
plt.show()


 
Practical 5
Cmd installation:
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
Code:
# Import necessary libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import svm 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score  

# Load the breast cancer dataset 
from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer() 
X = data.data 
y = data.target 

# Standardize the features 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# Perform PCA to reduce to 2D for visualization purposes 
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled) 

# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42) 

# Function to plot the decision boundary for SVM 
def plot_decision_boundary(clf, X, y, title): 
    h = .02  # step size in the mesh 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k') 
    plt.title(title) 
    plt.xlabel('PCA Component 1') 
    plt.ylabel('PCA Component 2') 
    plt.show() 

# Function to print evaluation metrics (confusion matrix and classification report) 
def evaluate_model(clf, X_test, y_test, y_pred, title): 
    print(f"Evaluation for {title}:\n") 
    
    # Accuracy, F1 Score, Precision 
    accuracy = accuracy_score(y_test, y_pred) 
    f1 = f1_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred) 

    print(f"Accuracy: {accuracy:.4f}") 
    print(f"F1 Score: {f1:.4f}") 
    print(f"Precision: {precision:.4f}") 

    # Confusion matrix 
    cm = confusion_matrix(y_test, y_pred) 
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues') 
    plt.title(f'Confusion Matrix - {title}') 
    plt.xlabel('Predicted Label') 
    plt.ylabel('True Label') 
    plt.show() 

    # Classification report 
    print(f"Classification Report for {title}:\n") 
    print(classification_report(y_test, y_pred)) 

# Applying Linear SVM 
linear_svm = svm.SVC(kernel='linear') 
linear_svm.fit(X_train, y_train) 
y_pred_linear = linear_svm.predict(X_test) 

# Plot Linear SVM 
plot_decision_boundary(linear_svm, X_test, y_test, 'Linear SVM (After Classification)') 
evaluate_model(linear_svm, X_test, y_test, y_pred_linear, 'Linear SVM') 

# Applying Non-linear SVM with RBF Kernel 
rbf_svm = svm.SVC(kernel='rbf') 
rbf_svm.fit(X_train, y_train) 
y_pred_rbf = rbf_svm.predict(X_test) 

# Plot RBF Kernel SVM 
plot_decision_boundary(rbf_svm, X_test, y_test, 'Non-linear SVM with RBF Kernel (After Classification)') 
evaluate_model(rbf_svm, X_test, y_test, y_pred_rbf, 'RBF Kernel SVM')
 
Practical 6
Cmd installation:
pip install numpy pandas matplotlib seaborn scikit-learn
pip install scikit-learn --upgrade
Code:
# Import necessary libraries for visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
import warnings

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load Spambase dataset from UCI (via fetch_openml)
spam_data = fetch_openml(name='spambase', version=1)

# Convert to DataFrame
df = pd.DataFrame(data=spam_data.data, columns=spam_data.feature_names)
df['target'] = spam_data.target.astype(int)

# Print the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Distribution of target (spam vs not spam)
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Distribution of Target (Spam vs Not Spam)')
plt.xlabel('Target (0 = Not Spam, 1 = Spam)')
plt.ylabel('Count')
plt.show()

# Features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a weak classifier (decision stump)
weak_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost model using the weak classifier with the SAMME algorithm
ada_boost = AdaBoostClassifier(estimator=weak_classifier, n_estimators=50, random_state=42, algorithm='SAMME')

# Train the AdaBoost model
ada_boost.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ada_boost.predict(X_test)

# Evaluate the performance of the AdaBoost model
print("Classification Report for AdaBoost:")
print(classification_report(y_test, y_pred))
ada_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (AdaBoost):", ada_accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix - AdaBoost Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Calculate ROC AUC Score for AdaBoost
y_pred_proba_ada = ada_boost.predict_proba(X_test)[:, 1]
roc_auc_ada = roc_auc_score(y_test, y_pred_proba_ada)
print("ROC AUC Score for AdaBoost:", roc_auc_ada)

# Compare with weak classifier (decision stump)
weak_classifier.fit(X_train, y_train)
y_pred_weak = weak_classifier.predict(X_test)
y_pred_proba_weak = weak_classifier.predict_proba(X_test)[:, 1]

# Evaluate weak classifier
print("Classification Report for Weak Classifier (Decision Stump):")
print(classification_report(y_test, y_pred_weak))
weak_accuracy = accuracy_score(y_test, y_pred_weak)
print("Accuracy (Weak Classifier):", weak_accuracy)

# Confusion Matrix for Weak Classifier
conf_matrix_weak = confusion_matrix(y_test, y_pred_weak)

# Plot Confusion Matrix Heatmap for Weak Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_weak, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix - Weak Classifier (Decision Stump)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Calculate ROC AUC Score for Weak Classifier
roc_auc_weak = roc_auc_score(y_test, y_pred_proba_weak)
print("ROC AUC Score for Weak Classifier:", roc_auc_weak)

# Plot ROC Curves for both AdaBoost and Weak Classifier together
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_proba_ada)
fpr_weak, tpr_weak, _ = roc_curve(y_test, y_pred_proba_weak)

plt.figure(figsize=(8, 6))
plt.plot(fpr_ada, tpr_ada, color='blue', label=f'AdaBoost (AUC = {roc_auc_ada:.2f})')
plt.plot(fpr_weak, tpr_weak, color='green', label=f'Weak Classifier (AUC = {roc_auc_weak:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - AdaBoost vs Weak Classifier')
plt.legend(loc="lower right")
plt.show()

# Plot a bar graph comparing the accuracy of AdaBoost and Weak Classifier
plt.figure(figsize=(6, 4))
models = ['AdaBoost', 'Weak Classifier']
accuracies = [ada_accuracy, weak_accuracy]
sns.barplot(x=models, y=accuracies, hue=models, palette='Set1', legend=False)  # Added hue parameter
plt.title('Accuracy Comparison: AdaBoost vs Weak Classifier')
plt.ylabel('Accuracy')
plt.show()
 
Practical 7
Cmd installation:
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install seaborn
Code:
# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import GaussianNB, BernoulliNB 
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.datasets import load_iris 
import seaborn as sns  

# Importing the dataset from sklearn 
iris = load_iris() 
X = iris.data[:, [0, 3]]  # Selecting features (sepal length and petal width) 
y = iris.target            # Selecting target variable (species) 

# Splitting the dataset into the Training set and Test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

# Feature Scaling (only for GaussianNB) 
sc = StandardScaler() 
X_train_scaled = sc.fit_transform(X_train) 
X_test_scaled = sc.transform(X_test) 

# Training the Gaussian Naive Bayes model on the Training set 
gaussian_classifier = GaussianNB() 
gaussian_classifier.fit(X_train_scaled, y_train) 

# Predicting the Test set results for Gaussian 
y_pred_gaussian = gaussian_classifier.predict(X_test_scaled) 

# Evaluating the Gaussian Naive Bayes model 
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian) 
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian) 

# Training the Bernoulli Naive Bayes model on the Training set (no scaling) 
bernoulli_classifier = BernoulliNB() 
bernoulli_classifier.fit(X_train, y_train) 

# Predicting the Test set results for Bernoulli 
y_pred_bernoulli = bernoulli_classifier.predict(X_test) 

# Evaluating the Bernoulli Naive Bayes model 
accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli) 
cm_bernoulli = confusion_matrix(y_test, y_pred_bernoulli) 

# Print the results 
print("Gaussian Naive Bayes:") 
print("Predicted Test Results: ", y_pred_gaussian) 
print("Model Accuracy: ", accuracy_gaussian * 100, "%") 
print("Confusion Matrix:\n", cm_gaussian) 
print("~" * 20) 

print("Bernoulli Naive Bayes:") 
print("Predicted Test Results: ", y_pred_bernoulli) 
print("Model Accuracy: ", accuracy_bernoulli * 100, "%") 
print("Confusion Matrix:\n", cm_bernoulli) 
print("~" * 20) 

# Visualization of Accuracy Comparison 
labels = ['Gaussian', 'Bernoulli'] 
accuracies = [accuracy_gaussian * 100, accuracy_bernoulli * 100] 

# Create a colorful bar plot for accuracy 
colors = ['#FF9999', '#99FF99'] 
plt.figure(figsize=(10, 6)) 
bars = plt.bar(labels, accuracies, color=colors) 

plt.title('Accuracy Comparison of Naive Bayes Classifiers') 
plt.xlabel('Classifier') 
plt.ylabel('Accuracy (%)') 
plt.ylim(0, 100) 

# Adding data labels to bars 
for bar in bars: 
    yval = bar.get_height() 
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center') 

plt.show() 

# Function to plot confusion matrix 
def plot_confusion_matrix(cm, title): 
    plt.figure(figsize=(6, 5)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False) 
    plt.title(title) 
    plt.ylabel('Actual') 
    plt.xlabel('Predicted') 
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=iris.target_names) 
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=iris.target_names, rotation=0) 
    plt.show() 

# Plot confusion matrices for each classifier 
plot_confusion_matrix(cm_gaussian, 'Gaussian Naive Bayes Confusion Matrix') 
plot_confusion_matrix(cm_bernoulli, 'Bernoulli Naive Bayes Confusion Matrix')
 
Practical 8
Cmd installation:
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
Code:
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the Breast Cancer dataset 
cancer = load_breast_cancer() 
X = cancer.data 
y = cancer.target 

# Convert to DataFrame for better visualization 
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names) 

# Display the first 8 rows of the dataset 
print("\nFirst 8 rows of the dataset:") 
print(df_cancer.head(8)) 

# Display the first 8 rows of the target column 
print("\nFirst 8 rows of the target column:") 
print(pd.Series(cancer.target).head(8)) 

# Data Preprocessing: Standardize the features 
scaler = StandardScaler() 
X = scaler.fit_transform(X) 

# Visualizing the dataset before classification using PCA 
pca = PCA(n_components=2)  # Reducing to 2 components 
X_pca = pca.fit_transform(X) 

# Create a DataFrame for visualization 
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2']) 
df_pca['target'] = y 

# Plot the dataset before classification 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='PCA1', y='PCA2', hue='target', data=df_pca, palette='Set1') 
plt.title('Breast Cancer Dataset Visualization (Before Classification)') 
plt.show() 

# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# Transform the data using PCA for visualization 
X_train_pca = pca.fit_transform(X_train) 
X_test_pca = pca.transform(X_test) 

# Implement the K-NN algorithm for different k-values 
k_values = [2, 5, 7, 9] 
for k in k_values: 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train) 
    y_pred = knn.predict(X_test) 
    
    # Evaluate the accuracy of the predictions 
    accuracy = accuracy_score(y_test, y_pred) 
    print(f'\nAccuracy for k={k}: {accuracy:.2f}') 
    
    # Display the classification report 
    print(f"\nClassification Report for k={k}:") 
    print(classification_report(y_test, y_pred, target_names=cancer.target_names)) 
    
    # Confusion matrix 
    conf_matrix = confusion_matrix(y_test, y_pred) 
    
    # Visualize the confusion matrix 
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cancer.target_names, yticklabels=cancer.target_names) 
    plt.xlabel('Predicted Label') 
    plt.ylabel('True Label') 
    plt.title(f'Confusion Matrix for k={k}') 
    plt.show() 

    # Create a mesh grid for plotting decision boundaries 
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1 
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), 
                         np.arange(y_min, y_max, 0.02)) 
    Z = knn.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])) 
    Z = Z.reshape(xx.shape) 

    # Plot decision boundaries 
    plt.figure(figsize=(10, 6)) 
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1') 
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k', cmap='Set1') 
    plt.xlabel('PCA Component 1') 
    plt.ylabel('PCA Component 2') 
 
    plt.title(f'Decision Boundaries for k={k}') 
    plt.show() 

# Cross-validation to find the optimal number of neighbors 
neighbors = np.arange(1, 21) 
cv_scores = [] 

# Perform 10-fold cross-validation 
for k in neighbors: 
    knn_cv = KNeighborsClassifier(n_neighbors=k) 
    scores = cross_val_score(knn_cv, X_train, y_train, cv=10, scoring='accuracy') 
    cv_scores.append(scores.mean()) 

# Plotting error rate vs. k value 
plt.figure(figsize=(10, 6)) 
plt.plot(neighbors, 1 - np.array(cv_scores), marker='o', linestyle='--', color='b') 
plt.xlabel('Number of Neighbors K') 
plt.ylabel('Error Rate') 
plt.title('Error Rate vs. K Value') 
plt.show() 

# Define k values for KNN and store accuracy scores 
k_range = range(1, 21) 
accuracy_scores = [] 

for k in k_range: 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train) 
    y_pred = knn.predict(X_test) 
    
    # Evaluate the accuracy of the predictions 
    accuracy = accuracy_score(y_test, y_pred) 
    accuracy_scores.append(accuracy) 

# Print the k values and their corresponding accuracy scores 
print("\nK values and their accuracy:") 
for k, acc in zip(k_range, accuracy_scores): 
    print(f'k={k}: Accuracy={acc:.2f}') 

# Plotting accuracy vs. k value 
plt.figure(figsize=(10, 6)) 
plt.plot(k_range, accuracy_scores, marker='o', linestyle='-', color='b', label='Accuracy') 
plt.xlabel('Number of Neighbors (k)') 
plt.ylabel('Accuracy') 
plt.title('Accuracy vs. Number of Neighbors (k)') 
plt.xticks(k_range) 
plt.grid(True) 
plt.legend() 
plt.show()
 
Practical 9
Cmd installation:
pip install pandas numpy mlxtend matplotlib seaborn
Code:
# Import necessary libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Create a random dataset
num_samples = 1000
data = {
    'id': range(num_samples),
    'full_name': [f'Person {i}' for i in range(num_samples)],
    'age': np.random.randint(15, 100, size=num_samples),
    'bmi': np.random.uniform(15, 40, size=num_samples),
    'blood_pressure': np.random.randint(60, 180, size=num_samples),
    'glucose_levels': np.random.randint(70, 300, size=num_samples),
    'gender': np.random.choice(['Male', 'Female'], size=num_samples),
    'smoking_status': np.random.choice(['Smoker', 'Non-smoker'], size=num_samples),
    'condition': np.random.choice(['Diabetic', 'Hypertensive', 'Healthy'], size=num_samples)
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Drop unnecessary columns
df = df.drop(columns=['id', 'full_name'])

# Bin continuous variables into categories
df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
df['bmi_bin'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['blood_pressure_bin'] = pd.cut(df['blood_pressure'], bins=[0, 80, 120, 140, 200], labels=['Low', 'Normal', 'Prehypertension', 'Hypertension'])
df['glucose_levels_bin'] = pd.cut(df['glucose_levels'], bins=[0, 90, 140, 200, 300], labels=['Normal', 'Pre-diabetic', 'Diabetic', 'High Diabetic'])

# Convert categorical columns to one-hot encoded variables
df = pd.get_dummies(df, columns=['gender', 'smoking_status', 'condition', 'age_bin', 'bmi_bin', 'blood_pressure_bin', 'glucose_levels_bin'])

# Drop original continuous columns
df.drop(columns=['age', 'bmi', 'blood_pressure', 'glucose_levels'], inplace=True)

# Print dataset after preprocessing
print("\nPreprocessed Data: \n", df.head())

# Set display options to avoid squeezing the output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Step 3: Apply the Apriori Algorithm to find frequent itemsets with minimum support of 0.1 (10%)
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

# Limit the frequent itemsets display to the first 10 rows
print("\nFrequent Itemsets (First 10 rows): \n", frequent_itemsets.head(10))

# Step 4: Generate Association Rules with minimum confidence of 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display only the first 10 association rules
print("\nAssociation Rules (First 10 rows): \n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Evaluate and interpret the rules
# Top 10 rules based on confidence
top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
print("\nTop 10 Association Rules: \n", top_rules)

# --- Visualization of Support and Confidence ---
# Plot support vs confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(x="support", y="confidence", size="lift", hue="lift", data=rules, palette="coolwarm", sizes=(40, 200))
plt.title("Association Rules - Support vs Confidence")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.legend(loc='upper right')
plt.show()

# Plot lift for top 10 rules (modifying to remove the warning)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_rules['lift'], y=top_rules.index, hue=top_rules['lift'], palette='viridis', legend=False)
plt.title("Top 10 Rules - Lift")
plt.xlabel("Lift")
plt.ylabel("Rule Index")
plt.show()

# --- New Addition: Plot top 10 rules by support ---
top_rules_by_support = rules.sort_values(by='support', ascending=False).head(10)

# Plot support for top 10 rules
plt.figure(figsize=(8, 6))
sns.barplot(x=top_rules_by_support['support'], y=top_rules_by_support.index, hue=top_rules_by_support['support'], palette='Blues', legend=False)
plt.title("Top 10 Rules - Support")
plt.xlabel("Support")
plt.ylabel("Rule Index")
plt.show()

 
Practical 10
Cmd installation:
Pip install tensorflow
Pip install tensorflow matplotlib
pip install opencv-python numpy
Code: 
First we have to run this code in idle file:
import os
import cv2
import numpy as np

# Create directories for dataset
os.makedirs('random_digits/train/0', exist_ok=True)
os.makedirs('random_digits/train/1', exist_ok=True)
os.makedirs('random_digits/train/2', exist_ok=True)
os.makedirs('random_digits/train/3', exist_ok=True)
os.makedirs('random_digits/train/4', exist_ok=True)
os.makedirs('random_digits/train/5', exist_ok=True)
os.makedirs('random_digits/train/6', exist_ok=True)
os.makedirs('random_digits/train/7', exist_ok=True)
os.makedirs('random_digits/train/8', exist_ok=True)
os.makedirs('random_digits/train/9', exist_ok=True)

# Function to generate random digit images
def generate_random_digit_images(num_images=1000):
    for _ in range(num_images):
        digit = np.random.randint(0, 10)  # Random digit from 0 to 9
        # Create a blank image
        img = np.ones((28, 28), dtype=np.uint8) * 255
        # Set the font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw the digit on the image
        cv2.putText(img, str(digit), (5, 20), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # Save the image in the corresponding folder
        cv2.imwrite(f'random_digits/train/{digit}/{_}.png', img)

# Generate random digit images
generate_random_digit_images()

Then after running above code cut the code and
Then run the following code in the same idle file:
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Image parameters
img_width, img_height = 28, 28
batch_size = 32
epochs = 5

# Directory for the dataset
train_data_dir = 'random_digits/train'

# Load the dataset using ImageDataGenerator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='sparse'
)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(img_width, img_height)),  # Flatten the 28x28 images
    tf.keras.layers.Dense(128, activation='relu'),   # First hidden layer
    tf.keras.layers.Dropout(0.2),                    # Dropout layer to reduce overfitting
    tf.keras.layers.Dense(10)                         # Output layer with 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs)

# Visualize some predictions
def display_predictions(generator, n=10):
    images, labels = next(generator)
    predictions = model.predict(images)

    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Pred: {predictions[i].argmax()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display predictions for the first 10 generated images
display_predictions(train_generator)


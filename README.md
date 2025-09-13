# CODSOFT Internship - Data Science (Batch B47)

This repository contains all the tasks and projects completed during my **CodSoft Data Science Internship** (20th August 2025 – 20th September 2025).

---

## 👩‍💻 Intern Details
- **Name:** Kolapaka Ramya Sri
- **College:** JNTUH College of Engineering, Jagityala
- **Batch:** August B47
- **Domain:** Data Science

---

## 📂 Repository Structure
- `Task-1/` → Source code + README + screenshots
- `Task-2/` → Source code + README + screenshots
- `Task-3/` → Source code + README + screenshots
- `Extras/` → Certificates, demo video links, notes

---

## 🚀 Tasks
### 🔹 Task 1: Student Performance Analysis
- **Description:**
- In this task, a dataset of student marks in three subjects (Maths, Science, English) was created and analyzed using Pandas and NumPy. Descriptive statistics were computed, and a bar chart was plotted using Matplotlib to visualize subject-wise performance. This helps in identifying variations in student performance across subjects.
- **Tech Stack:** Python, Pandas, Numpy, Matplotlib
- **Output Screenshot:**  
  ![WhatsApp Image 2025-09-12 at 8 04 18 PM](https://github.com/user-attachments/assets/02659f86-e102-4d79-a5df-5294c03cd950)
- **Code:**
- import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "Student": ["A", "B", "C", "D", "E"],
    "Maths": [78, 85, 96, 80, 72],
    "Science": [88, 92, 79, 94, 86],
    "English": [90, 76, 85, 83, 89]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)
print("\nSummary Statistics:\n", df.describe())

df.plot(x="Student", y=["Maths", "Science", "English"], kind="bar")
plt.title("Student Performance in Subjects")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("task1_output.png")
plt.show()


### 🔹 Task 2: Iris Dataset Classification
- **Description:**
- In this task, the famous Iris dataset is used for classification. The dataset contains measurements of iris flowers (sepal length, sepal width, petal length, petal width) and the goal is to classify them into three species: Setosa, Versicolor, and Virginica. A machine learning model is trained using Scikit-learn, and visualizations are created using Seaborn.
- **Tech Stack:** Python, Scikit-learn, Seaborn
- **Output Screenshot:**  
  ![WhatsApp Image 2025-09-13 at 1 15 13 PM](https://github.com/user-attachments/assets/9b3a7452-b2c1-4208-b5a7-5e03f11f25f7)
  - **Code**
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
iris_df = sns.load_dataset("iris")
sns.pairplot(iris_df, hue="species")
plt.show()



### 🔹 Task 3: House Price Prediction
- **Description:**
- In this task, a simple regression model is built to predict house prices based on features like area and number of bedrooms. A sample dataset is created, and the model is trained using Linear Regression from Scikit-learn. Finally, predictions and a visualization are generated.
- **Tech Stack:** Python, ML Libraries
- **Output Screenshot:**  
![WhatsApp Image 2025-09-13 at 1 23 57 PM](https://github.com/user-attachments/assets/4d3cf1b5-1fcc-4818-9252-61a4f621a199)
- **Code:**
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {
    "Area": [1200, 1400, 1600, 1800, 2000, 2200],
    "Bedrooms": [2, 3, 3, 4, 4, 5],
    "Price": [200000, 250000, 270000, 300000, 320000, 350000]
}
df = pd.DataFrame(data)
X = df[["Area", "Bedrooms"]]
y = df["Price"]
model = LinearRegression()
model.fit(X, y)
predicted_price = model.predict([[2100, 4]])
print("Predicted Price for Area=2100 & Bedrooms=4:", predicted_price[0])
plt.scatter(df["Area"], df["Price"], color="blue", label="Actual Prices")
plt.plot(df["Area"], model.predict(df[["Area", "Bedrooms"]]), color="red", label="Regression Line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.legend()
plt.show()



---

## 🎥 Demo Videos
I will share demo videos of each task on **LinkedIn** (tagged @CodSoft, #codsoft).  
Links are also saved in `Extras/demo_video_links.txt`.

---

## 📜 Certificates
- Offer Letter ✔
- Completion Certificate (To be updated after internship completion)

---

## 🔗 Connect
- **LinkedIn:** https://linkedin.com/in/kolapaka-ramya-sri
- **GitHub:** https://github.com/ramya-187

import pandas as pd

# --- STEP 1: LOAD AND INSPECT ---
# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Check for any missing/blank values
print("Checking for missing values:")
print(df.isnull().sum())

# --- STEP 2: SEPARATE FEATURES AND TARGET ---
# X gets all the soil and weather data (drops the crop name)
X = df.drop('label', axis=1) 

# y gets ONLY the crop name (the answers)
y = df['label']
# --- STEP 3: LABEL ENCODING ---
from sklearn.preprocessing import LabelEncoder

# Create the encoder
encoder = LabelEncoder()

# Transform the text crop names into numbers
y_encoded = encoder.fit_transform(y)

# Let's print it to see the magic happen!
print("\nOriginal Text Labels:")
print(y.head()) # Shows the first 5 text names

print("\nEncoded Number Labels:")
print(y_encoded[:5]) # Shows the first 5 numbers
# --- STEP 4: TRAIN / TEST SPLIT ---
from sklearn.model_selection import train_test_split

# Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"\nTotal dataset size: {len(X)} rows")
print(f"Training data size: {len(X_train)} rows")
print(f"Testing data size: {len(X_test)} rows")

# --- STEP 5: TRAIN AND TEST THE ML MODEL ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Create the model (Building the "committee" of 100 trees)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train the model (The "Studying" phase using the 80% data)
model.fit(X_train, y_train)

# 3. Test the model (Taking the "Final Exam" using the hidden 20% data)
predictions = model.predict(X_test)

# 4. Grade the exam (How many crops did it guess correctly?)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# --- STEP 6: SAVE THE MODEL AND ENCODER ---
import pickle

# 1. Save the trained Random Forest model
with open('crop_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# 2. Save the Label Encoder (so we can translate numbers back to text later!)
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

print("\nModel and Encoder successfully saved as .pkl files!")
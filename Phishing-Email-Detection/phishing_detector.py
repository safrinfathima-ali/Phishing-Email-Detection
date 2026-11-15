
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

data = {
    "text": [
        "Congratulations! You won a lottery. Click here to claim your prize.",
        "Urgent! Verify your bank account immediately at http://secure-login.xyz",
        "Your account has been suspended. Confirm your password now.",
        "You have won a free gift card. Claim your reward today.",
        "Update your payment details immediately to avoid account closure.",
        "Click here to receive your cash reward.",
        "Your package delivery failed. Pay the fee at http://delivery-check.tk",
        "Your email account will be closed. Verify your identity now.",
        "Congratulations! You are selected for a free iPhone.",
        "Unusual login detected. Reset your password at http://login-verify.ml",
        "You have an unpaid invoice. Open the attached payment link.",
        "URGENT: Your bank account is blocked. Click here now!",
        "Your monthly salary slip is attached for review.",
        "Hello, are we still meeting for lunch tomorrow?",
        "Please find the meeting agenda attached.",
        "Your order has been shipped and is arriving tomorrow.",
        "Thank you for attending today's meeting.",
        "The project report is ready for your review.",
        "Your appointment is confirmed for Monday morning.",
        "Please submit your assignment before Friday.",
        "The office will remain closed during the holidays.",
        "Your electricity bill is available in your account.",
        "Here are the notes from our previous discussion.",
        "Your subscription payment was successfully received."
    ],
    "label": [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

def extract_features(text):
    text = text.lower()
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    suspicious_words = ["urgent", "verify", "password", "claim", "winner", "prize", "suspended", "blocked"]
    suspicious_tlds = [".xyz", ".tk", ".ml", ".top", ".click"]

    keyword_count = sum(text.count(word) for word in suspicious_words)
    suspicious_url = int(any(any(tld in url for tld in suspicious_tlds) for url in urls))

    return text + " " + " ".join(["url_present"] * len(urls)) + " " + " ".join(["suspicious_keyword"] * keyword_count) + " " + ("suspicious_domain " if suspicious_url else "")

df["processed_text"] = df["text"].apply(extract_features)

X = df["processed_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("\nPHISHING EMAIL DETECTION MODEL")
print("--------------------------------")
print(f"Dataset size: {len(df)} emails")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
display.plot(cmap="Blues")
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nTest Your Own Email")
email = input("Enter email text: ")

processed_email = extract_features(email)
email_vectorized = vectorizer.transform([processed_email])
prediction = model.predict(email_vectorized)[0]

if prediction == 1:
    print("\nResult: PHISHING EMAIL")
else:
    print("\nResult: SAFE EMAIL")

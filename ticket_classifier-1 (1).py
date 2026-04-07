
# # 🎫 Customer Support Ticket Classifier
# **Automatically classify tickets by category and assign priority levels**
# 
# ---
# **Skills covered:** Text preprocessing · NLP classification · Priority logic · Model evaluation
# 

# ## Step 1 — Import Libraries

# In[1]:
import re
import string
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import joblib

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

print('✅ All libraries imported successfully!')

# ## Step 2 — Create the Dataset

# In[2]:
data = {
    'ticket_id': range(1, 51),
    'text': [
        # Billing (10)
        'My payment failed and I was charged twice for the same order',
        'I need a refund for the subscription I cancelled last week',
        'Wrong amount charged on my invoice from this month',
        'I was billed for a premium plan but I only signed up for basic',
        'My credit card was declined but the charge still went through',
        'Invoice number 4521 is missing from my account history',
        'I need to update my billing address and payment method',
        'Charged twice in the same month, please investigate',
        'My annual subscription renewed without warning or consent',
        'Tax invoice is showing incorrect amounts for last quarter',

        # Technical (10)
        'App crashes immediately every time I try to open it on iPhone',
        'The entire server is down, none of our 200 employees can work',
        'Critical production bug causing data loss in customer records',
        'Dashboard charts and graphs are not loading at all',
        'All users have been locked out after the latest system update',
        'Export to PDF feature is completely broken and returns an error',
        'Login page gives a 500 error for all users in our organization',
        'API is returning incorrect data and breaking our integrations',
        'Mobile app is extremely slow and times out on every screen',
        'Two-factor authentication is not sending SMS codes to any user',

        # Account (10)
        'Cannot log into my account after resetting my password yesterday',
        'My account has been hacked, please help secure it urgently',
        'I need to transfer ownership of the account to my colleague',
        'How do I add a new admin user to my team account',
        'Account was suspended without any warning or explanation',
        'I need to change the email address associated with my account',
        'How do I delete my account and all personal data permanently',
        'I forgot my username and cannot access the password reset page',
        'My profile information is not saving when I try to update it',
        'Two users in my team have duplicate accounts that need merging',

        # Shipping (10)
        'My order has been delayed by two weeks with no status update',
        'Wrong item was delivered to my address, I ordered something else',
        'Package arrived severely damaged and contents are broken',
        'Tracking number shows delivered but I never received the package',
        'Order was shipped to the wrong address despite correct checkout',
        'I need to change my delivery address for an order placed today',
        'Missing items from my order, only half the products arrived',
        'Express shipping was charged but item shipped via standard',
        'Return label is not working and I cannot send the item back',
        'Order cancelled by the system but my card was already charged',

        # General (10)
        'How do I export all my data to a CSV file',
        'Can you explain the difference between the Pro and Basic plans',
        'I would love to request a dark mode feature for the dashboard',
        'How do I integrate your service with Slack notifications',
        'What are your support hours and response time guarantees',
        'How do I generate a monthly usage report for my team',
        'Is there a mobile app available for Android devices',
        'How do I set up automated email notifications for my account',
        'Where can I find documentation for your REST API endpoints',
        'Can I get an extension on my trial period for evaluation',
    ],
    'category': (
        ['billing'] * 10 +
        ['technical'] * 10 +
        ['account'] * 10 +
        ['shipping'] * 10 +
        ['general'] * 10
    ),
    'priority': [
        # Billing
        'high', 'medium', 'high', 'high', 'high',
        'medium', 'low', 'high', 'medium', 'medium',
        # Technical
        'high', 'high', 'high', 'medium', 'high',
        'medium', 'high', 'high', 'medium', 'high',
        # Account
        'high', 'high', 'medium', 'low', 'high',
        'low', 'medium', 'medium', 'low', 'low',
        # Shipping
        'medium', 'medium', 'high', 'high', 'high',
        'medium', 'high', 'medium', 'medium', 'high',
        # General
        'low', 'low', 'low', 'low', 'low',
        'low', 'low', 'low', 'low', 'low',
    ]
}

df = pd.DataFrame(data)
df.to_csv('tickets.csv', index=False)

print(f'Dataset shape: {df.shape}')
print(f'\nCategory distribution:')
print(df['category'].value_counts())
print(f'\nPriority distribution:')
print(df['priority'].value_counts())
df.head()

# ## Step 3 — Visualise the Dataset

# In[3]:
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Dataset Overview', fontsize=14, fontweight='bold')

# Category distribution
cat_counts = df['category'].value_counts()
axes[0].bar(cat_counts.index, cat_counts.values,
            color=['#00C2FF', '#A259FF', '#FF6B6B', '#FFB800', '#00E5A0'])
axes[0].set_title('Tickets by Category')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')
for i, v in enumerate(cat_counts.values):
    axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

# Priority distribution
pri_counts = df['priority'].value_counts()
colors_pri = {'high': '#FF6B6B', 'medium': '#FFB800', 'low': '#00E5A0'}
bar_colors = [colors_pri[p] for p in pri_counts.index]
axes[1].bar(pri_counts.index, pri_counts.values, color=bar_colors)
axes[1].set_title('Tickets by Priority')
axes[1].set_xlabel('Priority')
axes[1].set_ylabel('Count')
for i, v in enumerate(pri_counts.values):
    axes[1].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ## Step 4 — Text Preprocessing

# In[4]:
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Full NLP preprocessing pipeline."""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove numbers
    text = re.sub(r'\d+', '', text)
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4. Tokenize
    tokens = nltk.word_tokenize(text)
    # 5. Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # 6. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

# Show before vs after
print('Preprocessing examples:\n')
for i in [0, 10, 20]:
    print(f'BEFORE: {df["text"].iloc[i]}')
    print(f'AFTER : {df["clean_text"].iloc[i]}')
    print()

# ## Step 5 — Word Cloud Visualisation

# In[5]:
try:
    from wordcloud import WordCloud

    categories = df['category'].unique()
    fig, axes = plt.subplots(1, len(categories), figsize=(18, 3))
    fig.suptitle('Most Common Words per Category', fontsize=13, fontweight='bold')

    palette = ['#00C2FF', '#A259FF', '#FF6B6B', '#FFB800', '#00E5A0']

    for ax, cat, color in zip(axes, categories, palette):
        text = ' '.join(df[df['category'] == cat]['clean_text'])
        wc = WordCloud(width=300, height=200,
                       background_color='white',
                       colormap='viridis',
                       max_words=30).generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(cat.upper(), color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()
except ImportError:
    print('wordcloud not installed. Run: pip install wordcloud')
    print('Skipping word cloud visualisation.')

# ## Step 6 — Feature Extraction with TF-IDF

# In[6]:
# Encode labels
cat_encoder = LabelEncoder()
pri_encoder = LabelEncoder()

y_cat = cat_encoder.fit_transform(df['category'])
y_pri = pri_encoder.fit_transform(df['priority'])

# TF-IDF vectorisation
tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),  # unigrams + bigrams
    sublinear_tf=True    # dampens very frequent terms
)
X = tfidf.fit_transform(df['clean_text'])

print(f'Feature matrix shape : {X.shape}')
print(f'Categories            : {list(cat_encoder.classes_)}')
print(f'Priorities            : {list(pri_encoder.classes_)}')

# Train / test split
X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
    X, y_cat, y_pri, test_size=0.2, random_state=42, stratify=y_cat
)

print(f'\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}')

# ## Step 7 — Train & Compare Multiple Models

# In[7]:
models = {
    'Naive Bayes':          MultinomialNB(),
    'Logistic Regression':  LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (LinearSVC)':      LinearSVC(random_state=42, max_iter=2000),
}

cat_results  = {}
pri_results  = {}

print('=' * 55)
print(f'{'Model':<25} {'Category Acc':>14} {'Priority Acc':>14}')
print('=' * 55)

for name, model in models.items():
    # Category model
    model.fit(X_train, y_cat_train)
    cat_pred = model.predict(X_test)
    cat_acc  = accuracy_score(y_cat_test, cat_pred)
    cat_results[name] = {'model': model, 'pred': cat_pred, 'acc': cat_acc}

    # Priority model (re-instantiate to avoid state sharing)
    import copy
    pri_model = copy.deepcopy(model)
    pri_model.fit(X_train, y_pri_train)
    pri_pred = pri_model.predict(X_test)
    pri_acc  = accuracy_score(y_pri_test, pri_pred)
    pri_results[name] = {'model': pri_model, 'pred': pri_pred, 'acc': pri_acc}

    print(f'{name:<25} {cat_acc:>13.1%} {pri_acc:>13.1%}')

print('=' * 55)

# Pick best models
best_cat_name = max(cat_results, key=lambda k: cat_results[k]['acc'])
best_pri_name = max(pri_results, key=lambda k: pri_results[k]['acc'])
print(f'\n✅ Best category model : {best_cat_name}  ({cat_results[best_cat_name]["acc"]:.1%})')
print(f'✅ Best priority model  : {best_pri_name}  ({pri_results[best_pri_name]["acc"]:.1%})')

# ## Step 8 — Model Evaluation

# In[8]:
# Detailed reports for best models
best_cat_pred = cat_results[best_cat_name]['pred']
best_pri_pred = pri_results[best_pri_name]['pred']

print('=== CATEGORY — Classification Report ===')
print(classification_report(y_cat_test, best_cat_pred,
                             target_names=cat_encoder.classes_))

print('=== PRIORITY — Classification Report ===')
print(classification_report(y_pri_test, best_pri_pred,
                             target_names=pri_encoder.classes_))

# In[9]:
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrices — Best Models', fontsize=13, fontweight='bold')

for ax, y_true, y_pred, labels, title in [
    (axes[0], y_cat_test, best_cat_pred, cat_encoder.classes_, f'Category ({best_cat_name})'),
    (axes[1], y_pri_test, best_pri_pred, pri_encoder.classes_, f'Priority ({best_pri_name})'),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.show()

# In[10]:
# Accuracy bar chart — all models
model_names = list(models.keys())
cat_accs = [cat_results[n]['acc'] for n in model_names]
pri_accs = [pri_results[n]['acc'] for n in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, cat_accs, width, label='Category', color='#00C2FF', alpha=0.85)
bars2 = ax.bar(x + width/2, pri_accs, width, label='Priority',  color='#A259FF', alpha=0.85)

ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ## Step 9 — Build the Classifier System

# In[11]:
class TicketClassifier:
    """
    End-to-end customer support ticket classifier.
    Predicts category and priority from raw ticket text.
    """

    PRIORITY_ACTIONS = {
        'high':   '🔴 Escalate immediately — SLA: respond within 1 hour',
        'medium': '🟡 Assign to next available agent — SLA: 4 hours',
        'low':    '🟢 Add to general queue — SLA: 24 hours',
    }

    CATEGORY_TEAMS = {
        'billing':   '💳 Billing & Payments Team',
        'technical': '🔧 Engineering / Technical Support',
        'account':   '👤 Account Management Team',
        'shipping':  '📦 Logistics & Fulfilment Team',
        'general':   '📋 General Support Team',
    }

    def __init__(self):
        self.tfidf       = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
        self.cat_model   = LogisticRegression(max_iter=1000, random_state=42)
        self.pri_model   = LogisticRegression(max_iter=1000, random_state=42)
        self.cat_encoder = LabelEncoder()
        self.pri_encoder = LabelEncoder()
        self._trained    = False

    def train(self, df: pd.DataFrame):
        """Train both classifiers on the provided DataFrame."""
        df = df.copy()
        df['clean'] = df['text'].apply(preprocess)

        X     = self.tfidf.fit_transform(df['clean'])
        y_cat = self.cat_encoder.fit_transform(df['category'])
        y_pri = self.pri_encoder.fit_transform(df['priority'])

        self.cat_model.fit(X, y_cat)
        self.pri_model.fit(X, y_pri)
        self._trained = True
        print('✅ TicketClassifier trained successfully!')

    def predict(self, ticket_text: str) -> dict:
        """Classify a single ticket. Returns a result dict."""
        if not self._trained:
            raise RuntimeError('Call .train() before .predict()')

        clean    = preprocess(ticket_text)
        vec      = self.tfidf.transform([clean])

        category = self.cat_encoder.inverse_transform(self.cat_model.predict(vec))[0]
        priority = self.pri_encoder.inverse_transform(self.pri_model.predict(vec))[0]

        # Confidence scores
        try:
            cat_proba = self.cat_model.predict_proba(vec)[0].max()
            pri_proba = self.pri_model.predict_proba(vec)[0].max()
            confidence = f'{(cat_proba + pri_proba) / 2:.0%}'
        except AttributeError:
            confidence = 'N/A'

        return {
            'ticket':     ticket_text,
            'category':   category,
            'priority':   priority,
            'team':       self.CATEGORY_TEAMS.get(category, 'General Support'),
            'action':     self.PRIORITY_ACTIONS.get(priority, 'Review manually'),
            'confidence': confidence,
        }

    def predict_batch(self, tickets: list) -> pd.DataFrame:
        """Classify a list of tickets and return a DataFrame."""
        return pd.DataFrame([self.predict(t) for t in tickets])

    def save(self, path='ticket_classifier.pkl'):
        joblib.dump(self, path)
        print(f'✅ Model saved → {path}')

    @staticmethod
    def load(path='ticket_classifier.pkl'):
        clf = joblib.load(path)
        print(f'✅ Model loaded ← {path}')
        return clf


# ---- Train the classifier ----
classifier = TicketClassifier()
classifier.train(df)

# ## Step 10 — Test the Classifier on New Tickets

# In[12]:
new_tickets = [
    'My account is completely locked and nobody on my team can log in',
    'I was charged three times for a single order this morning',
    'The production API is down and causing failures across all services',
    'How do I change the language setting in my dashboard?',
    'My package was delivered to the wrong address and I need it urgently',
]

results = classifier.predict_batch(new_tickets)

print('\n' + '=' * 80)
for _, row in results.iterrows():
    print(f"\n📩 TICKET   : {row['ticket']}")
    print(f"   Category  : {row['category'].upper()}  →  {row['team']}")
    print(f"   Priority  : {row['priority'].upper()}")
    print(f"   Action    : {row['action']}")
    print(f"   Confidence: {row['confidence']}")
print('\n' + '=' * 80)

# ## Step 11 — Analytics Dashboard

# In[13]:
# Classify the entire training dataset and show analytics
all_results = classifier.predict_batch(df['text'].tolist())

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Support Ticket Analytics Dashboard', fontsize=14, fontweight='bold')

palette = ['#00C2FF', '#A259FF', '#FF6B6B', '#FFB800', '#00E5A0']

# 1 — Predicted category breakdown
cat_counts = all_results['category'].value_counts()
axes[0, 0].pie(cat_counts.values, labels=cat_counts.index,
               autopct='%1.0f%%', colors=palette, startangle=90)
axes[0, 0].set_title('Tickets by Category')

# 2 — Priority breakdown
pri_counts = all_results['priority'].value_counts()
pri_colors = [{'high': '#FF6B6B', 'medium': '#FFB800', 'low': '#00E5A0'}[p]
              for p in pri_counts.index]
axes[0, 1].bar(pri_counts.index, pri_counts.values, color=pri_colors, edgecolor='white')
axes[0, 1].set_title('Tickets by Priority Level')
axes[0, 1].set_ylabel('Count')
for i, v in enumerate(pri_counts.values):
    axes[0, 1].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

# 3 — Category × Priority heatmap
pivot = pd.crosstab(all_results['category'], all_results['priority'])
# Reorder priority columns if present
ordered_cols = [c for c in ['high', 'medium', 'low'] if c in pivot.columns]
pivot = pivot[ordered_cols]
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, ax=axes[1, 0])
axes[1, 0].set_title('Category × Priority Heatmap')

# 4 — High priority tickets per category
high_pri = all_results[all_results['priority'] == 'high']['category'].value_counts()
axes[1, 1].barh(high_pri.index, high_pri.values, color='#FF6B6B', edgecolor='white')
axes[1, 1].set_title('🔴 High Priority Tickets by Category')
axes[1, 1].set_xlabel('Count')
for i, v in enumerate(high_pri.values):
    axes[1, 1].text(v + 0.1, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ## Step 12 — Save & Reload the Model

# In[14]:
# Save
classifier.save('ticket_classifier.pkl')

# Reload and verify
loaded_clf = TicketClassifier.load('ticket_classifier.pkl')

test = loaded_clf.predict('Server is completely down, all users affected!')
print(f"\nVerification test:")
print(f"  Ticket   : {test['ticket']}")
print(f"  Category : {test['category']}")
print(f"  Priority : {test['priority']}")
print(f"  Action   : {test['action']}")
print('\n✅ Model saved and reloaded successfully!')

# ---
# ## 🎉 Project Complete!
# 
# | File | Description |
# |------|-------------|
# | `tickets.csv` | Labelled training dataset (50 tickets) |
# | `ticket_classifier.pkl` | Saved trained model |
# | This notebook | Full pipeline: data → preprocess → train → evaluate → deploy |
# 
# **To classify a new ticket in the future:**
# ```python
# clf = TicketClassifier.load('ticket_classifier.pkl')
# result = clf.predict('Your new ticket text here')
# print(result)
# ```
# 
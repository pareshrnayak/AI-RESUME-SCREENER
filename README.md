# 🎯 AI Resume Screener
(AI RESUME SCREENING SYSTEM)

The **AI Resume Screener** is a web-based application designed to simplify and automate the initial stage of candidate screening. It allows users to upload resumes and compare them against a given job description, helping identify the most relevant candidates quickly and efficiently.

---

## 🚀 Overview

This project uses **Natural Language Processing (NLP)** techniques to analyze and compare resumes with a job description. It calculates how well each candidate matches the role and ranks them accordingly.

The system also highlights **matched skills**, identifies **missing skills**, and provides **visual insights** to support better decision-making in the hiring process.

---

## 🛠️ Tech Stack

### 🔹 Backend
- **Python**
- **Flask**

### 🔹 Frontend
- **HTML**
- **CSS**
- **JavaScript**

### 🔹 Libraries & Tools
- **scikit-learn** → TF-IDF & Cosine Similarity
- **NLTK** → Text processing
- **PyPDF2** → PDF parsing
- **Chart.js** → Data visualization

---

## 🧠 Techniques Used

- **TF-IDF (Term Frequency – Inverse Document Frequency)**  
  Converts text into numerical vectors based on word importance.

- **Cosine Similarity**  
  Measures similarity between job description and resumes.

- **Text Preprocessing (NLP)**  
  Cleaning, tokenization, and normalization of text data.

---

## ⚙️ How It Works

1. User enters a **Job Description**
2. Uploads up to **5 resumes** (PDF or TXT)
3. System extracts and processes text from resumes
4. Applies **TF-IDF vectorization**
5. Calculates **cosine similarity scores**
6. Ranks candidates based on match percentage
7. Displays:
   - Candidate ranking
   - Matched skills
   - Missing skills
   - Visual charts

---

## ✨ Features

- 📄 Upload up to **5 resumes** (PDF/TXT)
- 📌 Input **Job Description**
- 🧠 NLP-based resume analysis
- 🏆 **Candidate ranking** based on match score
- ✅ Shows **matched skills**
- ❌ Identifies **missing skills**
- 📊 **Bar chart** for score comparison
- 🧠 **Radar chart** for top candidate skill distribution
- ⬇️ Download analysis as **HTML report**
- 📦 Export results in **JSON format**
- 🎨 Clean and interactive UI

---

## 📸 Screenshots
<img width="1918" height="1018" alt="DEMO1" src="https://github.com/user-attachments/assets/b6edb459-131e-497d-8405-1f092815bf50" />

<img width="1918" height="1021" alt="DEMO2" src="https://github.com/user-attachments/assets/2f4f7a16-a53d-4815-a24c-aa18276003ec" />

<img width="1918" height="916" alt="DEMO3" src="https://github.com/user-attachments/assets/256ee7ab-0c58-40ff-9113-c1c88b797668" />

<img width="1918" height="1015" alt="DEMO4" src="https://github.com/user-attachments/assets/5fc27f04-b287-4882-8c37-e69784f04b0d" />

<img width="1918" height="1013" alt="DEMO5" src="https://github.com/user-attachments/assets/c71deeb1-5c04-4304-a4ad-39e88e5dec6e" />


---

## ▶️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/pareshrnayak/AI-RESUME-SCREENER.git

# Navigate into project folder
cd AI-RESUME-SCREENER

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

Developed by Paresh Nayak & Karan Bhandary as part of Machine Learning Internship at Swizosoft (OPC) Pvt. Ltd

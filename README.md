This project includes a complete pipeline for job title classification, grade prediction, and salary estimation using modern Natural Language Processing (NLP) and Machine Learning techniques.
The goal is to take a new job description and predict its job grade, estimate the salary, and match it to the closest reference job title. Although several methods were explored in the notebooks, the app uses only the best models for job title, grade, and salary prediction.



The structure of the folder is as follows:



Application/

│

├── Job\_Title/

│   └── job\_title\_classification.ipynb         │

├── Grade\&Salary/

│   ├── 1.grade\_salary\_prediction\_distilroberta.ipynb

│   ├── 2.grade\_salary\_prediction\_MiniLM+DNN.ipynb

│   ├── 3.grade\_salary\_prediction\_mpnet+DNN.ipynb

│   ├── 4.grade\_salary\_prediction\_pharaphraseminilm+DNN.ipynb

│   ├── 5.grade\_embedding+RF.ipynb

│   ├── 6.grade\_tf-idf+xgboost\_Rf.ipynb

│   ├── 7.salary\_embedding+xgboost\_RF.ipynb

│   └── 8.salary\_tf-idf+xgboost\&Rf.ipynb│

├── app.py            # Final Streamlit application

└── README.md



│


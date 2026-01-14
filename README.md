# Adversarial Machine Learning for Improved Cybersecurity Threat Detection

This research project investigates the vulnerability of machine learning-based intrusion detection systems to adversarial attacks. Using the NSL-KDD dataset, we evaluate how small perturbations — generated using the Fast Gradient Sign Method (FGSM) — impact model classification accuracy.

## 📘 Research Objectives
- Analyze the NSL-KDD dataset and construct baseline IDS models.
- Implement FGSM to craft adversarial samples.
- Evaluate the magnitude of accuracy degradation under targeted attacks.
- Compare model robustness (Random Forest vs Neural Network).
- Deploy a real-time analytical dashboard using Flask for attack visualization.

## 🧠 Methodology
1. Dataset preprocessing & feature encoding  
2. Baseline model training  
3. Gradient-based adversarial sample generation  
4. Post-attack accuracy evaluation  
5. Visualization of perturbation effects  

## 📊 Key Findings
- Neural Networks exhibit significantly higher vulnerability to FGSM attacks.
- Random Forest models show more stability due to ensemble behavior.
- Even minimal perturbations cause misclassification in deep learning models.
- Adversarial training improves robustness but increases training cost.

## 🔧 System Architecture
- Data Processing Pipeline  
- Model Training & Storage  
- Adversarial Attack Generator  
- Flask Dashboard with WebSocket updates  
- SQL-based authentication system  

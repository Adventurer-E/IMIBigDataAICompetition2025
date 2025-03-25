# Money Laundering Detection with Unsupervised Learning
This is a project completed for [IMI Big Data & AI Competition 2025](https://www.utm.utoronto.ca/bigdataaihub/events/sixth-annual-2024-2025-imi-bigdataaihub-big-data-and-artificial-intelligence-competition). It's a collaborated work with [Aayush Bhan](https://www.linkedin.com/in/aayush-bhan-681734215/), [Eric Lu](https://www.linkedin.com/in/eric-l-483375134/), Richard Xu and Chelsea Zhao.
## Overview
This project aims to detect potential money laundering activities using an unsupervised learning approach. Given a dataset of transactions and customers where money-laundering transactions are **not labeled**, we develop a feature engineering pipeline, leverage deep learning techniques for anomaly detection, and build a production-ready system for deployment.
## Approach
1. **Feature Engineering**: Manually create meaningful features that capture essential characteristics related to money laundering.
2. **Neural Network Embeddings**: Train a neural network on customer data to generate embeddings that encode transactional behavior.
3. **Anomaly Detection with Contrastive Learning**: Utilize contrastive learning techniques to identify unusual patterns in customer behavior, flagging potential fraudulent activities.
## Outputs
 - **Customer Embeddings**: High-dimensional representations of customers based on transaction behaviors.
 - **Anomaly Detection Model**: A trained model that can identify suspicious transactions.
 - **Production Pipeline**: A containerized workflow using Docker to ensure scalability and ease of deployment.
## Technologies Used
 - **Python** (Pandas, NumPy, Scikit-learn, PyTorch/TensorFlow)
 - **Neural Networks** for embedding generation
 - **Contrastive Learning** for anomaly detection
 - **Docker** for containerized deployment

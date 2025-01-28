# Customer Segmentation and Recommendation

This project is a Streamlit-based application that segments customers using a K-Means clustering model and provides investment recommendations.

## Features
- Upload a customer dataset (CSV format).
- Perform clustering using pre-trained models.
- Receive investment recommendations for each customer.
- Download the segmented results as a CSV file.

## Setup

### Prerequisites
- Python 3.9+
- Dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-segmentation.git

2. Navigate to the project directory
   ```bash
    cd customer-segmentation

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

Running the App
To run the app locally, use the following command:

   ```bash
streamlit run app.py


File Structure

-app.py: Main application logic.
-model/: Contains pre-trained models (kmeans.pkl).
-requirements.txt: Lists required Python packages.

"""Download the original IBM sample dataset; retain upstream attribution."""
from pathlib import Path
from urllib.request import urlretrieve
URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
if __name__ == '__main__':
    destination = Path(__file__).resolve().parent / 'data' / 'telco.csv'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        print('Dataset already exists:', destination)
    else:
        urlretrieve(URL, destination)
        print('Downloaded:', destination)

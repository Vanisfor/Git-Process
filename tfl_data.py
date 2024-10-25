import requests
import csv

# TfL API endpoint for BikePoint
url = "https://api.tfl.gov.uk/BikePoint/"

# Send a GET request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()
    
    # Prepare the CSV file
    csv_file = 'bikepoints_data.csv'
    csv_columns = ['Station', 'ID', 'Bikes Available', 'Docks Available']

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        for station in data:
            writer.writerow({
                'Station': station['commonName'],
                'ID': station['id'],
                'Bikes Available': station['additionalProperties'][6]['value'],
                'Docks Available': station['additionalProperties'][7]['value']
            })
    
    output_message = f"Data successfully saved to {csv_file}."
else:
    output_message = f"Failed to retrieve data. HTTP Status code: {response.status_code}"

output_message
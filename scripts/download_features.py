import requests
import os
import zipfile

# Get the latest artifact
api_url = f'https://api.github.com/repos/{os.environ["GITHUB_REPOSITORY"]}/actions/artifacts'
headers = {
    'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}',
    'Accept': 'application/vnd.github.v3+json'
}

response = requests.get(api_url, headers=headers)
response.raise_for_status()

# Find most recent aqi-features artifact
artifacts = [a for a in response.json()['artifacts'] 
            if a['name'].startswith('aqi-features')]

if not artifacts:
    raise Exception('No matching artifacts found')
    
latest = max(artifacts, key=lambda x: x['updated_at'])
print(f'Downloading artifact: {latest["name"]} (ID: {latest["id"]})')

# Download
os.makedirs('data', exist_ok=True)
download_url = latest['archive_download_url']
response = requests.get(download_url, headers=headers, stream=True)
response.raise_for_status()

zip_path = 'data/features.zip'
with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
        
# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('data/')
os.remove(zip_path)
print('Download and extraction completed successfully')

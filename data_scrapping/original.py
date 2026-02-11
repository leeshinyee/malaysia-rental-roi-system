import json
from flask import Flask, request, jsonify
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape():
    # Get payload from Node-RED
    payload = request.json

    url = payload.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400
    # Initialize the Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    # Open the webpage
    driver.get(url)
    # Wait for the page to load
    driver.implicitly_wait(10)
    # Get the page source
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    # Find the button with 'data-id'
    button = soup.find('button', {'data-id': True})
    ad_id = button.get('data-id') if button else None
    # Extract JSON data
    script_tag = soup.find('script', id='__NEXT_DATA__')
    json_data = json.loads(script_tag.string)

    ad_details = json_data['props']['initialState']['adDetails']['byID'][ad_id]['attributes']
    category_params = ad_details['categoryParams']
    property_params = ad_details['propertyParams']

    # Extract required information
    ad_info = {}
    for param in category_params:
        if param['id'] == 'monthly_rent':
            ad_info['monthly_rent'] = param['value']
        if param['id'] == 'category_id':
            ad_info['property_type'] = param['value']
        if param['id'] == 'location':
            ad_info['location'] = param['value']
        if param['id'] == 'rooms':
            ad_info['room_quantity'] = param['value']
        if param['id'] == 'bathroom':
            ad_info['bathroom'] = param['value']
        if param['id'] == 'size':
            ad_info['size'] = param['value']
        if param['id'] == 'furnished':
            ad_info['furnished'] = param['value']
        if param['id'] == 'facilities':
            ad_info['facilities'] = param['value']
        if param['id'] == 'additional_facilities':
            ad_info['additional_facilities'] = param['value']
        if param['id'] == 'parking':
            ad_info['parking'] = param['value']
        if param['id'] == 'region':
            ad_info['region'] = param['value']

    # Extract building details
    for param_group in property_params:
        if param_group['header'] == 'BUILDING DETAILS':
            for param in param_group['params']:
                if param['id'] == 'prop_name':
                    ad_info['property_name'] = param['value']
                if param['id'] == 'completion_year':
                    ad_info['completion_year'] = param['value']

    # Close the browser
    driver.quit()

    return jsonify(ad_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

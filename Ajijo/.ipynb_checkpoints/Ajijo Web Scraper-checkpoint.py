# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 00:44:05 2020

@author: Abhinav Pandey

Title - Ajijo Data Scraper
"""

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from IPython.display import clear_output

# Cleaning Functions
def clean_tag(ptag):
    return ptag.get('onclick').replace("window.location.href=", "")
def clean_price(price):
    return price.get('value').replace("Add 1 for" ,"")

# Extracting price and product info from ONE page
def page_extractor(url):
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    
    prod_info = soup.find_all('div', class_="product-info")
    price_info = soup.find_all('input', class_="btn", type='button')
    
    products = [prod.find('a').text for prod in prod_info]
    prices = [clean_price(price) for price in price_info]
    price_item_tags = [clean_tag(ptag) for ptag in price_info]
        
    category = url.replace("https://www.ajijo.com.au/collections/", "")
    category_feature = [category] * len(products)

    data = pd.DataFrame({'name' : products,
                         'price_correspondance' : price_item_tags,
                         'collection' : category_feature,
                         'price' : prices})
    
    return data

# Extracting price and product info from ALL pages
def website_extractor():

    url = "https://www.ajijo.com.au/collections/basmati-rice"
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')

    # Get all product catgeories from Basmati Rice webpage
    all_collections = [ele.get('href') for ele in soup.find_all('a', href=True) if ("/collections/" in ele.get('href')) & ("/products/" not in ele.get('href'))]
    all_collections = np.unique(all_collections)
    
    # Compile urls
    all_urls = ["https://www.ajijo.com.au" + ele for ele in all_collections]
    
    # Extract data from each url
    all_data = pd.DataFrame() # Initialize Empty DataFrame
    for i, url in enumerate(all_urls): 
        
        clear_output(wait=True)
        
        page_data = page_extractor(url)    
        all_data = pd.concat((all_data, page_data))
        
        print("Product and Price Data Compilation Progress : {} %".format(np.round((i+1)/len(all_urls)*100 ,2)))
        
        
    all_data.reset_index(drop=True, inplace=True)

    return all_data

# Extract inventory Data
def extract_inventory_data(data):
    
    url_base = "https://www.ajijo.com.au/collections/"

    data['item_url'] = url_base + data['collection'] + data['price_correspondance']
    data['item_url'] = data['item_url'].str.replace("'","")

    bad_url_condition = data.item_url.str.contains("page")
    data.loc[bad_url_condition, 'item_url'] = data.loc[bad_url_condition, 'item_url'].str.replace("\?page=2", "")
    data.loc[bad_url_condition, 'item_url'] = data.loc[bad_url_condition, 'item_url'].str.replace("\?page=3", "")

    inventory = []
    url_list = []
    urls = data['item_url']

    for i, url in enumerate(urls):

        clear_output(wait=True)

        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        inventory_info = soup.find_all('div', class_="inventory")[0].text

        inventory.append(inventory_info)
        url_list.append(url)

        print("Inventory Data Compilation Progress : {} %".format(np.round(((i+1)/len(data)*100) ,2)))

    inventory_data = pd.DataFrame({'inventory' : inventory,
                                    'item_url' : url_list})

    inventory_data = inventory_data[~inventory_data.duplicated()]

    return inventory_data

# Clean Inventory Data
def clean_inventory_data(inventory_data):

        case_1 = inventory_data['inventory'].str.count("\n") == 9 
        case_2 = (inventory_data['inventory'].str.count("\n") == 5) & (inventory_data['inventory'].str.contains("available!"))
        case_3 = inventory_data['inventory'].str.contains("out of stock")

        inventory_data.loc[case_1, 'inventory'] = inventory_data.loc[case_1, 'inventory'].apply(lambda x : x.replace("\n","").split()[3])
        inventory_data.loc[case_2, 'inventory'] = -99 # Available but unknown
        inventory_data.loc[case_3, 'inventory'] = -199 # Out of Stock

        inventory_data.inventory = inventory_data.inventory.astype(int)

        return inventory_data
    
    
def compile_dataset():
    
    price_product_data = website_extractor()
    inventory_data = extract_inventory_data(price_product_data)
    
    inventory_data = clean_inventory_data(inventory_data)
    
    dataset = pd.merge(price_product_data, inventory_data, on='item_url')
    dataset = dataset[['name', 'inventory', 'price', 'price_correspondance', 'collection', 'item_url']]
    
    return dataset

ajijo_dataset = compile_dataset()

ajijo_dataset.to_csv("data.csv", index=False)
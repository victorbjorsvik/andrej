import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re

def get_norwegian_male_names(url):
  """
  This function fetches the HTML content from the given URL and extracts all the Norwegian male names.

  Args:
      url: The URL of the webpage containing the list of Norwegian male names.

  Returns:
      A list of Norwegian male names extracted from the webpage.
  """
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')

  # Find all the table rows containing names (assuming they are within a table)
  link_list_items = soup.find_all('li', class_='link-list__item')

  pattern = r'\s*\(.*?\)'

  # Extract the names from the first table data cell (assuming that's where the names are)
  names = []
  for row in link_list_items:
      name = row.find('a').get_text()
      name = re.sub(pattern, '', name)
      if name:
        names.append(name)

  # Return the list of extracted names
  return names

# Example usage
url = "https://snl.no/.taxonomy/4024"  # Replace with the actual URL
boy_names = get_norwegian_male_names(url)

url2 = "https://snl.no/.taxonomy/4025"  # Replace with
girl_names = get_norwegian_male_names(url2)

with open('boy_names.txt', 'w') as f:
    for line in boy_names:
        f.write(f"{line}\n")

with open('girl_names.txt', 'w') as f:
    for line in girl_names:
        f.write(f"{line}\n")
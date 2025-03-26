import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def g(stores):
    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--window-size=10x10")

    driver = webdriver.Chrome(chrome_options=chrome_options, executable_path="/Users/quocsinh09/Downloads/chromedriver")
    
    df = pd.DataFrame()
    for store in stores:
        cmts = []
        try:
            url = "https://www.foody.vn/" + store+ "/binh-luan"
            driver.get(url)
            els = driver.find_elements(by=By.CSS_SELECTOR, value=".list-reviews .foody-box-review .pn-loadmore")

            while len(els) > 0:
                els[0].click()
                time.sleep(1)
                els = driver.find_elements(by=By.CSS_SELECTOR, value=".list-reviews .foody-box-review .pn-loadmore")

            cmts = driver.find_elements(by=By.CSS_SELECTOR, value=".rd-des")
            
        except:
            print("Error : " + store)

        if stores.index(store) % 100 == 0:
            print(store+ " : " + str(stores.index(store)))

        cmts = [el.text for el in cmts if el.text != '']
        cmts = list(dict.fromkeys(cmts))
        
        _dict = {"id":store, "comments":cmts}  
       
        df = df.append(pd.DataFrame(_dict), ignore_index=True) 

    return df

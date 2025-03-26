import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
    
def f(prn):
    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--window-size=10x10")

    driver = webdriver.Chrome(chrome_options=chrome_options, executable_path="/Users/quocsinh09/Downloads/chromedriver")
    
    stores_url = "https://www.foody.vn/"+prn+"/o-dau"

    driver.get(stores_url)
    time.sleep(3)
    els = driver.find_elements(by=By.CSS_SELECTOR, value=".btn-load-more")
    
    stores = []
    
    while len(els) > 0 :
        els[0].click()
        stores = []
        els = driver.find_elements(by=By.CSS_SELECTOR, value=".ldc-items-list .ldc-item .ldc-item-header .ldc-item-h-name h2 a")
        for i in els:
            stores.append(i.get_attribute('href').replace("https://www.foody.vn/", '')) if i.get_attribute('href') is not None else ''

        stores = list(dict.fromkeys(stores))
        if  len(stores) > 3000: break

        els = driver.find_elements(by=By.CSS_SELECTOR, value=".btn-load-more")
    
    return stores


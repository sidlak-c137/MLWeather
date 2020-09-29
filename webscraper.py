'''
Frank Hou
CSE 163 AC

This code produces a .csv file containing data parsed from the wunderground
website for all dates between 1950 and 2020 (present year). Note that the code
needs specification of a file path to the chrome browser before initializing
the webdriver.

This file is meant for local reproduction; however implementation details
locally could be overly complicated.

Requires download of the Selenium, pandas, sys, and time packages.
'''

from selenium import webdriver
import pandas
import sys
import time


def setup():
    '''
    Sets up the chromedriver, returns a working chromedriver to parse sites
    NOTE: This section will need editing for local-use, specifically putting a
    valid path to the chrome application.
    '''
    sys.path.insert(0, '/usr/lib/chromium-browser/chromedriver')
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    # Note: Indicate chrome path here
    wd = webdriver.Chrome('chromedriver', chrome_options=chrome_options)
    return wd


def add_to_dict(dictionary, string, key):
    '''
    Used privately by the code; assumes that the given parameters are all valid
    Valid indicates the first parameter is a dictionary, the second is the
    values in a "x y z" form, and the third parameter is a valid weather point.
    A valid weather point is either "temp", "humid", "wind", or "pressure".
    Given a dictionary, value string, and key, adds to the dictionary based
    on the key's max, min, avg and the given three values.
    '''
    holder = string.split()
    tup = ("_max", "_avg", "_min")
    for i in range(3):
        dictionary[key + tup[i]].append(float(holder[i]))


def get_data(data, year, month, driver):
    '''
    Given a data as a dictionary, a year value, and a month value, finds the
    weather in seattle relative to the given year and month, then stores the
    parsed data into the dictionary. Assume the dictionary is a mapping from
    string to a list.
    '''
    driver.get(
        "https://www.wunderground.com/history" +
        "/monthly/us/wa/seattle/KSEA/date/{}-{}-7"
        .format(year, month))
    time.sleep(3)
    tables = driver.find_elements_by_tag_name("table")
    # Time (Month), Temperature, Dew Point, Humidity, Wind Speed, Pressure
    days = tables[3].text.split()[1:]
    # Days table reading is a little broken on the scraper side; there's
    # occasionally an extra "1" at the end So we check for that and remove
    # if there is
    if days[len(days)-1] == '1':
        days = days[:len(days)-1]
    print(tables[4].text)
    temperature = tables[4].text.split("\n")
    print(temperature)
    humidity = tables[6].text.split("\n")
    wind = tables[7].text.split("\n")
    pressure = tables[8].text.split("\n")
    for i in range(1, len(days)+1):
        data["year"].append(year)
        data["month"].append(month)
        data["day"].append(days[i-1])
        add_to_dict(data, temperature[i], "temp")
        add_to_dict(data, humidity[i], "humid")
        add_to_dict(data, wind[i], "wind")
        add_to_dict(data, pressure[i], "pressure")


def main():
    wd = setup()
    data = dict()
    columns = ["year", "month", "day", "temp_max", "temp_avg", "temp_min",
               "humid_max", "humid_avg", "humid_min", "wind_max", "wind_avg",
               "wind_min", "pressure_max", "pressure_avg", "pressure_min"]
    for column in columns:
        data[column] = []

    for year in range(1950, 2021):
        for month in range(1, 13):
            try:
                get_data(data, year, month, wd)
            except Exception as e:
                print(e + ", Done (after 2020), or we ran into a problem!")
                break
    df = pandas.DataFrame(data)
    df.to_csv('data.csv', index=False)


if __name__ == "__main__":
    main()

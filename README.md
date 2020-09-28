# Weather Prediction
Created by Frank Hou and Sidharth Lakshmanan

This program will train and compare 3 different machine learning models that predict weather in Seattle
to determine which model is the best for this particular situation. By extension, the WeatherModels class will allow the client to tune each model, train each model, and use each model to predict future weather.
The models used are : Linear Regression Model, Decision Tree Regression Model, Neural Network
## Creating the .csv of the data
Implemented by Frank Hou
##### Running the Code in the Collab Notebook

* Run each code block in the Collab Notebook in sequence. This will produce **data.csv**, a file containing all of the weather data from the Observations section of:
    https://www.wunderground.com/history/monthly/us/wa/seattle/KSEA from 1950 to Present Day
This code will take around 10000 seconds to run.

* This file will be saved to your mounted drive (follow the instructions of the second code block for setting up your local drive). Make sure to change the drive path to the drive desired (in the last code block).

* Note: This may be able to run locally using the webscraper.py file provided if you are able to connect chromedriver to the web application (our local environments were unable to do this). To do this, you will have to specify the file path to the chrome browser. However, it is much more foolproof to run your code in the Collab notebook as it has been proven successful.

##### Testing the Code
To test the data parser, run the last block of code.
This will verify that the data was parsed correctly.

## Comparing, Training, and Tuning each model
Implemented by Sidharth Lakshmanan
##### Installation of Dependancies

* You must have all of the packages in the CSE 163 environment
* Use the following command to install altair saver  
        ```
        conda install -c conda-forge altair_saver
        ```
##### Running the Code
Note that some of the code in main.py is commented out. This code was used to create
the tuning graphs as seen in the **parameter_tunings** folder.
The second section of commented code creates a graph comparing each model, which is located in the
**model_comparison** folder.
Note:
**DO NOT MODIFY THE CONTENTS OF THESE 2 FOLDERS**

* To tune the parameters of the model, uncomment the code, change the folder path to "test", and run using:
        ```
        python main.py
        ```
        This will create html graphs in the "test" folder. To view the graphs,
        open the html documents in Google Chrome. From there, you will see an option
        (top left corner) to save the images in jpg format.
        *Note that this code will take ~5000 seconds to complete*
        **You may change the folder that the graphs are placed in, just remember to create a blank folder with the new name.**

* To produce the graphs of each model in comparison to one another, run using:
        ```
        python main.py
        ```
        This will create html a graph in the "test" folder. To view the graph,
        open the html document in Google Chrome. From there, you will see an option
        (top left corner) to save the image as a jpg format.
        *Note that this code will take ~1000 seconds to complete*
        **You may change the folder that the graphs are placed in, just remember to create a blank folder with the new name.**

##### Testing the WeatherModels class
To test the WeatherModels class, run the following command:
    ```
    python weather_models_test.py
    ```
This will verify the implementation of the WeatherModels class.
**Note: This will erase any preexisting graphs in the "test" folder**



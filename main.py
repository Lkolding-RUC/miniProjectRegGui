import pandas as pd
from DataClean import *
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tkinter import *
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Read csv files
vacdata = pd.read_csv('archive/country_vaccinations.csv')
popdata = pd.read_csv('archive/population_by_country_2020.csv')

# Rename columns in population dataset
popdata_new = popdata.rename(columns={'Country (or dependency)': 'country', 'Population (2020)': 'population'},
                             inplace=False)

# Datacleaning
clean_data_vac = DataClean(vacdata)
clean_data_pop = DataClean(popdata_new)

keeplistVac = ['country', 'date', 'people_fully_vaccinated']

keeplistPop = ['country', 'population']

clean_data_vac.keep_columns(keeplistVac)
clean_data_pop.keep_columns(keeplistPop)


# Group data
people_fully_vaccinated = vacdata.groupby(by=['country'], sort=False, as_index=False)['people_fully_vaccinated'].max()

for country in vacdata['country'].unique():
    vacdata.loc[vacdata['country'] == country, 'people_fully_vaccinated'] = interpolate_country(vacdata, country)

    
# Merge datasets
mergedata = pd.merge(vacdata, popdata_new)


# Function that returns the population of a specific country
def get_population(df, country):
    result = df.loc[df['country'] == country, 'population'].iloc[0]
    return result

# Function that returns the date of the start of the vaccination, of a specific country
def get_start_date(df, country):
    date_string = df.loc[df['country'] == country, 'date'].iloc[0]
    # Convert string to date
    result = dt.datetime.strptime(date_string, '%Y-%m-%d')
    return result

# Loop through 1, 2, 3 and 4th degree polynomials to find the best fit line based on r2 score. Select model with highest r2 value
def select_best_model(x, y):
    models = [2, 3, 4]
    model_number = 1
    model = np.poly1d(np.polyfit(x, y, model_number))
    score = r2_score(y, model(x))
    for i in models:
        model_2 = np.poly1d(np.polyfit(x, y, i))
        score_2 = r2_score(y, model_2(x))
        if score_2 > score:
            model = model_2
            score = score_2
            model_number = i
    print('Model with best certainty is: ', model_number)
    return model

# Function that runs when we choose a country
def predict(choice):
    selectedCountry = choice

    # Get population in a selected country
    selectedPop = get_population(popdata_new, choice)

    # Get the selected country's start date
    startDate = get_start_date(vacdata, choice)

    # Set specific country to be equal to choice in drop down menu
    spec_country = mergedata[mergedata.country == choice]
    
    # Set x to be equal to amount of days the specific country has been vaccinating
    spec_country['x'] = np.arange(len(spec_country))
    
    x = spec_country['x']
    y = spec_country['people_fully_vaccinated']

    # set model to be equal to best fit model found above & insert x, y values
    model = select_best_model(x, y)


    # Days from start day - to predicted day
    predictionDay = predict_fully_vaccinated_day(model, selectedPop)

    #the date where full vaccination is achieved
    fullyVaccinatedDay = startDate + dt.timedelta(predictionDay)

    numOfVacPeople = model(predictionDay)

    print('this is startDate', startDate)
    print('This is the population:', selectedPop)
    print('This is predictionDay', predictionDay)
    print('All in ', selectedCountry, ' will be vaccinated on:', fullyVaccinatedDay.date())
    print("At this day, this many people will be fully vaccinated: ", numOfVacPeople)

    print("We know this, with certainty from 0-1: ", r2_score(y, model(x)))

    # GUI
    pop_msg = Label(root, text="Population in your selected country : ", ).grid(row=9, column=0)
    pop_input = Label(root, text=selectedPop).grid(row=9, column=1)


    date_msg = Label(root, text="The final date where 100% of population will be vaccinated is: ").grid(row=10,column=0)
    date_input = Label(root, text=fullyVaccinatedDay.date()).grid(row=10, column=1)
    day_msg = Label(root, text="Amount of days from first vaccination: ").grid(row=11, column=0)
    day_input = Label(root, text=predictionDay).grid(row=11, column=1)


    line = np.linspace(0, 140, 100)  # last value = precision

    plt.scatter(x, y)
    plt.title('Current Regression Covid-19 vaccination')
    plt.xlabel('Days from first vaccination')
    plt.ylabel('Population given vaccine')
    plt.plot(line, model(line), color="red")
    plt.show()

    #Display plotted model in GUI
    figure = plt.Figure(figsize=(7, 6), dpi=100)
    ax = figure.add_subplot(111)
    ax.scatter(x, y)
    chart_type = FigureCanvasTkAgg(figure, root)
    chart_type.get_tk_widget().grid(row=13, column=0)
    ax.plot(line, model(line), color="red")

    ax.set_xlabel('Days from first vaccination')
    ax.set_ylabel('Population given vaccine')
    ax.set_title('Current Regression Covid-19 vaccination')

# måske en kommentar til denne også (men jeg kan ikke forklare det særlig godt)
def interpolate_country(df, country):
    firs = df.loc[df['country'] == country, 'people_fully_vaccinated'].index[0]
    col = df.columns.get_loc('people_fully_vaccinated')
    df.iloc[firs, col] = 0
    specific_col = 'people_fully_vaccinated'
    return df.loc[vacdata['country'] == country, specific_col].interpolate(limit_direction='both', limit=df.shape[0])


def predict_fully_vaccinated_day(model, population):
    dayCount = 0
    while dayCount < 5000:
        vaccinated = model(dayCount)
        if vaccinated > population:
            # Reached the population - return dayCount
            return dayCount
        dayCount = dayCount + 1
    return dayCount


# start GUI
root = Tk()
root.title("Corona vaccination prediction")
root.geometry("850x750")
clicked = StringVar(root)
clicked.set("Choose country")


message = Label(root, text="Pick a country below, and we'll predict when it will be fully vaccinated.")
drop_down_country = OptionMenu(root, clicked, *sorted(popdata_new.country), command=predict)

message.grid(row=1, column=0)
drop_down_country.grid(row=4, column=0)



root.mainloop()

import pandas as pd
from DataClean import *
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tkinter import *
import tkinter as tk


# Function that return the population of a specific country
def get_population(df, country):
    result = df.loc[df['country'] == country, 'population'].iloc[0]
    return result


# Function that return the date of the start of the vaccanation, of a specific country
def get_start_date(df, country):
    date_string = df.loc[df['country'] == country, 'date'].iloc[0]
    # Convert string to date
    result = dt.datetime.strptime(date_string, '%Y-%m-%d')
    return result


def select_best_model(x, y):
    models = [2, 3, 4]
    model_number = 1
    model = np.poly1d(np.polyfit(x, y, 1))
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


def predict(choice):
    selectedCountry = choice

    # Get population in a selected country
    selectedPop = get_population(popdata_new, choice)

    # Get the selected country's start date
    startDate = get_start_date(vacdata, choice)

    spec_country = mergedata[mergedata.country == choice]
    spec_country['x'] = np.arange(len(spec_country))

    # print('This is spec_country: \n', spec_country, 'here it ends')
    # print('This is the len of spec_country', len(spec_country))
    x = spec_country['x']
    y = spec_country['people_fully_vaccinated']

    model = select_best_model(x, y)
    # model = np.poly1d(np.polyfit(x, y, 5))
    line = np.linspace(0, 104, 100)  # last value = precision

    # Days from start day - to predicted day
    predictionDay = predict_fully_vaccinated_day(model, selectedPop)

    fullyVaccinatedDay = startDate + dt.timedelta(predictionDay)

    numOfVacPeople = model(predictionDay)

    print('this is startDate', startDate)
    print('This is the population:', selectedPop)
    print('This is predictionDay', predictionDay)
    print('All in ', selectedCountry, ' will be vaccinated on:', fullyVaccinatedDay.date())
    print("At this day, this many people will be fully vaccinated: ", numOfVacPeople)
    # print("The country has this many citizens: ", spec_country['population'])
    print("We know this, with certainty from 0-1: ", r2_score(y, model(x)))

    # GUI
    pop_msg = Label(root, text="Population in your selected country : ", ).grid(row=9, column=0)
    pop_input = Label(root, text=selectedPop).grid(row=9, column=1)

    # day_msg = Label(root, text="Days from first vaccination: ").grid(row=11, column=0)
    # day_input = Label(root, text=predictionDay).grid(row=11, column=1)
    date_msg = Label(root, text="The final date where 100% of population will be vaccinated is: ").grid(row=12,
                                                                                                        column=0)
    date_input = Label(root, text=fullyVaccinatedDay.date()).grid(row=12, column=1)

    # vac_msg = Label(root, text="At this day, this many people will be fully vaccinated: ").grid(row=13, column=0)
    # vac_input = Label(root, text=numOfVacPeople).grid(row=13, column=1)

    # model_msg = Label(root, text="The best fit model for this data is: ").grid(row=13, column=0)

    prediction = True

    if prediction == spec_country['population'].iloc[0]:
        print(y)

    plt.scatter(x, y)
    plt.plot(line, model(line), color="red")
    plt.show()


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


# Read csv files no paths pwease! Just keep data in separate data folder
vacdata = pd.read_csv('archive/country_vaccinations.csv')
popdata = pd.read_csv('archive/population_by_country_2020.csv')

# Rename columns in population dataset
popdata_new = popdata.rename(columns={'Country (or dependency)': 'country', 'Population (2020)': 'population'},
                             inplace=False)

clean_data_vac = DataClean(vacdata)
clean_data_pop = DataClean(popdata_new)

# Drops Items in dropList
dropListVac = ['iso_code', 'total_vaccinations', 'people_vaccinated', 'daily_vaccinations_raw',
               'daily_vaccinations', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
               'people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million', 'vaccines',
               'source_name', 'source_website']
dropListPop = ['Yearly Change', 'Net Change', 'Density (P/Km²)', 'Land Area (Km²)', 'Migrants (net)',
               'Fert. Rate', 'Med. Age', 'Urban Pop %', 'World Share']

clean_data_vac.removeCols(dropListVac)
clean_data_pop.removeCols(dropListPop)

# Group data
people_fully_vaccinated = vacdata.groupby(by=['country'], sort=False, as_index=False)['people_fully_vaccinated'].max()

for country in vacdata['country'].unique():
    vacdata.loc[vacdata['country'] == country, 'people_fully_vaccinated'] = interpolate_country(vacdata, country)

# start GUI
root = Tk()
root.title("Corona vaccination prediction")
root.geometry("700x200")
root.configure(bg='#E4E6E7')
clicked = StringVar(root)
clicked.set("Choose country")


message = Label(root, text="Pick a country below, and we'll predict when it will be fully vaccinated.")
drop_down_country = OptionMenu(root, clicked, *sorted(popdata_new.country), command=predict)

message.grid(row=1, column=0)
drop_down_country.grid(row=4, column=0)

# merge datasets
mergedata = pd.merge(vacdata, popdata_new)

root.mainloop()

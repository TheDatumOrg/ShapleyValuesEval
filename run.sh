#!/bin/bash

echo "strategy = $1"
methods_list=("logr mlp dt nb svm knn")

datasets_list=("
hepatitis
flag
audiology
Salaries
sales-records
50_Startups
laptop_price
drug200
hr-evaluation
anneal
wine
AirPassengers
500_Person_Gender_Height_Weight_Index
nursery
connect-4
hayes-roth
bank
heart-c
Mall_Customers
zoo
nba_logreg
"
"
soybean
drug_dt
seismic-bumps
melb_data
glass
ElectronicsData
results
adult
IBM_CHRN_CSV
Churn_Modelling
movie_data
phoneme
crime_data
ToyotaCorolla
Data_Uk
rainfall in india 1901-2015
jobs_in_data
deliveries
Churn Modelling Test Data
Titanic_Data
car
"
"
heart-statlog
credit-g
Car_sales
uforeports
FuelConsumptionCo2
churn
Bmi_male_female
wilt
insurance
haberman
echocardiogram
lymphography
covertype
tic-tac-toe
births
diabetes
imdrating
magic
healthcare-dataset-stroke-data
monk-3
census_data
"
"
pima-indians-diabetes
titanic
High Quality Dataset
poker-hand
employees
cleveland
advertising_classification
v3_Latest_Data_Science_Salaries
auto-mpg
car details v4
cars
bupa-liver-disorders
horse-colic
vote
iris
FuelConsumption
Housing_Data
Data_Preprocessing
CAR DETAILS FROM CAR DEKHO
"
"
car data
Cust_Segmentation
breast-cancer
sales-cars
SacramentocrimeJanuary2006
hungarian-heart-disease
airport-frequencies
ecoli
mushroom
Titanic_Test
weight-height
housing
cylinder-bands
segment
credit-a
air_pollution new
autos
Car details v3
balance-scale
sonar
monk-2
monk-1
airports
")


for method in $methods_list
    do
    for list in "${datasets_list[@]}"
    do
        number=$1
        for item in $list
        do        
            (taskset --cpu-list $number python main_algorithms.py --dataset $item --method $method --shap_method $2) &
            ((number++))
        done
        wait
    done
done

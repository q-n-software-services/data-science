import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''# Initializations
weight = 72.12      # In Kilograms exact
height = 5.75       # In feet and decimals obtained by dividing inches by 12
body_type = 'Asian'     # Regional identification for the body type
exercise_time = 12.12   # In Minutes and decimal obtained from dividing seconds by 60
exercise_days_streak = 5    # Number of days for which the consistency is achieved in exercise/workout
food_intake = 500.12        # Number of Calories, intake of food
exhaustion_level = 0.12     # In Percentage i.e. out of 1
physical_condition = 'Male Teenager'    # Gender and age group
'''

# https://download.data.world/file_download/bgadoci/crossfit-data/leaderboard.15.csv
# Above is the link from where the dataset of athletes' data is taken


# importing the Dataset
athlete_data = pd.read_excel("Athelete Data Filtered.xlsx")
age_data = athlete_data.iloc[:, 6].values
height_data = athlete_data.iloc[:, 7].values
weight_data = athlete_data.iloc[:, 8].values
pullups_data = athlete_data.iloc[:, 20].values
length = len(age_data)
'''
print(length)
print("all data\n", athlete_data)
print("age data", age_data)
print('height data', height_data)
print('weight data', weight_data)
print('pullups data', pullups_data)
'''

# *****   Clustering Algorithm     *****

# Initialisation
total_weight_of_all_athletes = 0.0
# Calculating BMI for all Athletes present in the Dataset
bmi = []
for i in range(length):
    this_bmi = (703 * weight_data[i])/(height_data[i] ** 2)
    bmi.append(this_bmi)


# Calculating Average Weight of an athlete
Mean_Weight = np.mean(weight_data)

print("Average Weight of an Athlete is :\t", Mean_Weight, " lb\t(Pounds)")

# Calculating Weight Standard Deviation of an athlete
SD_Weight = np.std(weight_data)
print("Standard Deviation in Weight of an Athlete is :\t", SD_Weight, " lb\t(Pounds)")

# Creating a DataFrame of BMI and Pullups-count of all the Athletes
df = pd.DataFrame({
    'x': bmi,
    'y': pullups_data


})

# Color Scheme for the Scatter Plot
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'c'}

# *****   K-Means Clustering Algorithm begins here     *****
np.random.seed(200)

# Number of clusters to be created
k = 5

# Generating Random points as centroids for starting the process
centroids = {
    i+1: [np.random.randint(0, 120), np.random.randint(0, 220)]
    for i in range(k)
}


# Assignment Stage, assignment of data-points to the respective cluster using the K-Means formula/Method

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 + (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df


df = assignment(df, centroids)
print(df)

# *****     Update Stage     *****

import copy

# Making a new dataset of previous centroids
old_centroids = copy.deepcopy(centroids)


def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k


centroids = update(centroids)

# Repeat Assignment Stage

df = assignment(df, centroids)

# Continue until all assigned categories don't change anymore

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

# Generating the Scatter Plot to see the Data-points and Respective Clusters
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 120)
plt.ylim(0, 220)
plt.show()


# Creating a list my_centroid to store the centroid coordinates for further processing
my_centroid = []
# Sorting the coordinates of centroids in ascending order with respect to 2nd value i.e. number of pullups performed by the athlete on daily basis
temp = []

for i in centroids.keys():
    my_centroid.append((centroids[i][0], centroids[i][1]))

for count in range(5):
    for counter in range(5):
        if my_centroid[count][1] < my_centroid[counter][1]:
            temp = my_centroid[counter]
            my_centroid[counter] = my_centroid[count]
            my_centroid[count] = temp

print("Centroids' coordinates are as follows")
for i in range(len(my_centroid)):
    print("Centroid ", i + 1, ":\t", my_centroid[i])

# Coordinates of centroids from our dataset.
centroid1 = [my_centroid[0][0], my_centroid[0][1]]
centroid2 = [my_centroid[1][0], my_centroid[1][1]]
centroid3 = [my_centroid[2][0], my_centroid[2][1]]
centroid4 = [my_centroid[3][0], my_centroid[3][1]]
centroid5 = [my_centroid[4][0], my_centroid[4][1]]
print()
print()

# Interacting with a new Entry from an anonymous person/user
# Input their details and intimating them about their fitness and relevant stuff
my_weight = float(input("Enter you weight in pounds(lb)"))
my_height = float(input("Enter your height in inches"))
my_bmi = (703 * my_weight)/(my_height ** 2)
my_pullups = float(input("Enter the Number of Pullups That you perform on daily basis"))

# Determining the cluster to which this new data-point would belong to
cent_dist = 200
nearby_centroid = None
count = 1

for i in centroids.keys():
    dist = np.sqrt(((centroids[i][0] - my_bmi) ** 2) + ((centroids[i][1] - my_pullups) ** 2))
    if dist <= cent_dist:
        nearby_centroid = centroids[i]
        cent_dist = dist

# This outputs the fitness group/ centroid to which the new user/person belongs
print("nearby centroid", nearby_centroid)

print()
print()

# Advising Exercise and food on the basis of weight od=f the respective person
if my_weight < (Mean_Weight - SD_Weight):
    calories_count = (Mean_Weight - my_weight) * 500
    if calories_count > 500:
        days = 7 + int(calories_count//500)
        daily_calories = int(calories_count / days)
    else:
        days = 7
        daily_calories = int(calories_count)

    print("You are under-weight, kindly increase your diet and amount of proteins to gain some weight\n"
          "I suggest you devise your daily diet by adding food that gives you an extra \t", daily_calories,
          "\t Calories \nfor atleast\t", days, " days\t inorder to gain weight uptill that of an athlete i.e.\t",
          Mean_Weight, ' lb\t (pounds)')

elif my_weight > (Mean_Weight + SD_Weight):
    print("You are  over-weight, kindly decrease your diet and amount of carbohydrates and fatty food to lose some weight.\n"
          "Moreover, do some additional exercise like hiking, sit-ups, pushups inorder to lose weight\n"
          "I Would suggest, \t", int(my_weight - Mean_Weight), " Pushups\tand\t", 2*int(my_weight - Mean_Weight), "Situps daily")
else:
    print("Your Weight is Optimum")

print()

# Following set of statements render Info to the person based on his/her fitness level and can be modified.
if nearby_centroid == centroid1: # Least Fitness level
    print("You have Poor Fitness")
    print("Either perform\t", centroid2[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid2[1] - centroid1[1])/15)*10, "\t Minutes each day")
    print("Inorder to gain the level of Average Fitness")

elif nearby_centroid == centroid2: # Level 2 Fitness
    print("You have Average Fitness")
    print("Either perform\t", centroid3[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid3[1] - centroid2[1]) / 15) * 10, "\t Minutes each day")
    print("Inorder to gain the level of Optimum Fitness")
elif nearby_centroid == centroid3: # Level 3 fitness
    print("You have Optimum Fitness")
    print("Either perform\t", centroid4[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid4[1] - centroid3[1]) / 15) * 10, "\t Minutes each day")
    print("Inorder to gain the level of Excellent Fitness")
elif nearby_centroid == centroid4: # Second Highest Fitness level
    print("You have Excellent Fitness")
    print("Either perform\t", centroid5[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid5[1] - centroid4[1]) / 15) * 10, "\t Minutes each day")
    print("Inorder to become a Fitness Freak")
elif nearby_centroid == centroid5: # Top of the most, Fitness level.
    print("You are a Fitness Freak")
    print("Keep up your exercise routine")





# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:33:38 2021

@author: jackw
"""
# Imports
import pandas as pd
import seaborn as sns
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
sns.set(font='Segoe UI', rc={"figure.facecolor":"#FF1801", "figure.dpi":300, 'savefig.dpi':300})

# Read in the F1 data
path1 = r"C:\Users\jackw\Downloads\archive (2)\AllRace.csv"
path2 = r"C:\Users\jackw\Downloads\archive (2)\ConstructorStandings.csv"
path3 = r"C:\Users\jackw\Downloads\archive (2)\DriversStandings.csv"

df_races = pd.DataFrame(pd.read_csv(path1))
df_constr = pd.DataFrame(pd.read_csv(path2))
df_driver = pd.DataFrame(pd.read_csv(path3))

#%% Fonts
'''
font_paths = mat.font_manager.findSystemFonts()
font_objects = mat.font_manager.createFontList(font_paths)
font_names = [f.name for f in font_objects]
print(font_names)
'''
#%% Create df with all driver names for later usage
"""""
df_drivers = pd.DataFrame(columns = ['Driver', 'Points'])
df_drivers = df_drivers.append({'Driver':drivers}, ignore_index=True)
df_drivers = df_drivers.explode('Driver')
df_drivers = df_drivers.set_index('Driver')
"""
#%% Cleaning Data

# Check for race df nulls
print(df_races['Date'].isnull().sum().sum())

# Remove .digits from the end of the race locations
df_races = df_races.replace(to_replace='\.\d*', value='', regex=True)

# Here we find two races with no winner results
print(df_races[df_races['1'].isnull()==True]) 

# Giving the winner of both Grand Prix
df_races.loc[379, '1'] = 'Michele Alboreto ALB' 
df_races.loc[648, '1'] = 'Michael Schumacher MSC'

# Check driver nulls
#print(df_driver[df_driver['Pos'].isnull()==True]) 
#print(df_driver[df_driver['Driver'].isnull()==True]) 
#print(df_driver[df_driver['Nationality'].isnull()==True]) 
print(df_driver[df_driver['Car'].isnull()==True]) # These Null values are for cars that did not have a name. I will not be adding values to these
#print(df_driver[df_driver['PTS'].isnull()==True])
#print(df_driver[df_driver['Year'].isnull()==True])

# Check constr nulls
print(df_constr.isnull().sum().sum())


#%% Who are the most dominant drivers and constructors

#%% Constructor Championships
df_constrWins = pd.DataFrame(df_constr[df_constr['Pos']=='1'])
df_constrWins = pd.DataFrame(df_constrWins['Team'].value_counts().reset_index())

constrWins_plt = sns.barplot(x='index', y='Team',data=df_constrWins, color='#FF1801')
for p in constrWins_plt.patches:
    constrWins_plt.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=9,
                   xytext = (0, 4), 
                   textcoords = 'offset points')
constrWins_plt.set_xticklabels(constrWins_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
constrWins_plt.tick_params(labelsize=11, colors='white')
constrWins_plt.set_title("Constructor Championships (1950 - 2020)", fontsize=22, color='white')
constrWins_plt.set_xlabel("Constructors", fontsize=19, color='white')
constrWins_plt.set_ylabel("Number of Championships", fontsize=19, color='white')
#%%
df_constrPoints = df_constr[df_constr['Team'].isin(['Ferrari', 'Mercedes', 'Lotus Ford'])]
fig, axes = plt.subplots()

df_constrPoints_plt = sns.violinplot(x='Team', y='PTS', data=df_constrPoints, ax=axes)
df_constrPoints_plt.set_title("Top Three Constructor Point Distribution (1950-2020)", fontsize=22, color='white')
df_constrPoints_plt.tick_params(labelsize=12, colors='white')
df_constrPoints_plt.set_xlabel("Constructor", fontsize=19, color='white')
df_constrPoints_plt.set_ylabel("Points", fontsize=19, color='white')
plt.show()

#%% Driver Championships
df_driverWins = pd.DataFrame(df_driver[df_driver['Pos']=='1'])
df_driverWins = pd.DataFrame(df_driverWins['Driver'].value_counts().reset_index())
df_driverWins = df_driverWins[df_driverWins['Driver'] > 1]

driverWins_plt = sns.barplot(x='index', y='Driver',data=df_driverWins, color='#FF1801')
for p in driverWins_plt.patches:
    driverWins_plt.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=9,
                   xytext = (0, 4), 
                   textcoords = 'offset points')
driverWins_plt.set_xticklabels(constrWins_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
driverWins_plt.tick_params(labelsize=12, colors='white')
driverWins_plt.set_title("Driver Championships (1950 - 2020)", fontsize=22, color='white')
driverWins_plt.set_xlabel("Drivers", fontsize=19, color='white')
driverWins_plt.set_ylabel("Number of Championships", fontsize=19, color='white')
#%%
df_driverPoints = df_driver[df_driver['Driver'].isin(['Michael Schumacher MSC', 'Lewis Hamilton HAM', 'Juan Manuel Fangio FAN', 'Sebastian Vettel VET', 'Alain Prost PRO'])]
fig, axes = plt.subplots()

df_driverPoints_plt = sns.violinplot(x='Driver', y='PTS', data=df_driverPoints, ax=axes)
df_driverPoints_plt.set_title("Top Five Driver Point Distribution (1950-2020)", fontsize=22,  color='white')
df_driverPoints_plt.set_xticklabels(df_driverPoints_plt.get_xticklabels(), rotation=22, horizontalalignment='right')
df_driverPoints_plt.tick_params(labelsize=13, colors='white')
df_driverPoints_plt.set_xlabel("Driver", fontsize=19, color='white')
df_driverPoints_plt.set_ylabel("Points", fontsize=19,  color='white')
plt.show()
#%% Who is the most dominant driver on the most difficult tracks
df_singapore = df_races[df_races['Unnamed: 0'] == 'Singapore']
df_monaco = df_races[df_races['Unnamed: 0'] == 'Monaco']
df_azerbaijan = df_races[df_races['Unnamed: 0'] == 'Azerbaijan']
df_japan = df_races[df_races['Unnamed: 0'] == 'Japan']

placements = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

#%% Most dominant driver in Singapore GP

# Purely wins
df_singaporeWins = pd.DataFrame(df_singapore['1'].value_counts().reset_index())

# Total points scored, (1st Place : 25 points, 2nd Place : 18, 3rd Place : 15})
df_singaporePlacements = pd.DataFrame(df_singapore.iloc[:, 2:12])

# Purely points scored
sing_drivers = []
for c in df_singaporePlacements.columns:
    counts = df_singaporePlacements[c].value_counts()
    for name, points in counts.items():
        if name not in sing_drivers:
            sing_drivers.append(name)
'''   
df_drivers = pd.DataFrame(columns=sing_drivers)
df_drivers.append(pd.Series(), ignore_index=True)
df1 = pd.DataFrame([[0] * len(df_drivers.columns)], columns=df_drivers.columns)
df_drivers = df1.append(df_drivers, ignore_index=True)
'''
df_sing_drivers = pd.DataFrame(columns = ['Driver', 'Points'])
df_sing_drivers = df_sing_drivers.append({'Driver':sing_drivers}, ignore_index=True)
df_sing_drivers = df_sing_drivers.explode('Driver')
df_sing_drivers = df_sing_drivers.set_index('Driver')

df_singaporePoints = df_sing_drivers.copy()
df_singaporePoints['Points'] = 0

"""
for c in df_singaporePlacements.columns:
    counts = df_singaporePlacements[c].value_counts()
    for name, points in counts.items():
        sing_points.at[name] = sing_points.at[name] + (counts[name] * placements[int(c)-1])
"""

for c in df_singaporePlacements.columns:
    counts = df_singaporePlacements[c].value_counts()
    for name, points in counts.items():
        df_singaporePoints.at[name, 'Points'] += (counts[name] * placements[int(c)-1])

df_singaporePoints = df_singaporePoints.reset_index()
df_singaporePoints = df_singaporePoints[df_singaporePoints['Points'] >= 10]
df_singaporePoints = df_singaporePoints.sort_values(['Points'])
#%% Singapore Plot
singaporepoints_plt = sns.barplot(x=df_singaporePoints['Points'], y=df_singaporePoints['Driver'], orient='h', color='#FF1801')

singaporepoints_plt.tick_params(labelsize=12, colors='white')
singaporepoints_plt.set_title("Points earned at Singapore Grand Prix by Driver", fontsize=22, color='white')
singaporepoints_plt.set_xlabel("Points", fontsize=19, color='white')
singaporepoints_plt.set_ylabel("Driver", fontsize=19, color='white')
#%% Most dominant driver in Monaco GP

# Purely wins
df_monacoWins = pd.DataFrame(df_monaco['1'].value_counts().reset_index())

# Total points scored, (1st Place : 25 points, 2nd Place : 18, 3rd Place : 15})
df_monacoPlacements = pd.DataFrame(df_monaco.iloc[:, 2:12])

# Purely points scored
monaco_drivers = []
for c in df_monacoPlacements.columns:
    counts = df_monacoPlacements[c].value_counts()
    for name, points in counts.items():
        if name not in monaco_drivers:
            monaco_drivers.append(name)


'''   
df_drivers = pd.DataFrame(columns=sing_drivers)
df_drivers.append(pd.Series(), ignore_index=True)
df1 = pd.DataFrame([[0] * len(df_drivers.columns)], columns=df_drivers.columns)
df_drivers = df1.append(df_drivers, ignore_index=True)
'''
df_monaco_drivers = pd.DataFrame(columns = ['Driver', 'Points'])
df_monaco_drivers = df_monaco_drivers.append({'Driver':monaco_drivers}, ignore_index=True)
df_monaco_drivers = df_monaco_drivers.explode('Driver')
df_monaco_drivers = df_monaco_drivers.set_index('Driver')

df_monacoPoints = df_monaco_drivers.copy()
df_monacoPoints['Points'] = 0

"""
for c in df_singaporePlacements.columns:
    counts = df_singaporePlacements[c].value_counts()
    for name, points in counts.items():
        sing_points.at[name] = sing_points.at[name] + (counts[name] * placements[int(c)-1])
"""

for c in df_monacoPlacements.columns:
    counts = df_monacoPlacements[c].value_counts()
    for name, points in counts.items():
        df_monacoPoints.at[name, 'Points'] += (counts[name] * placements[int(c)-1])

df_monacoPoints = df_monacoPoints.reset_index()
df_monacoPoints = df_monacoPoints[df_monacoPoints['Points'] >= 89]
df_monacoPoints = df_monacoPoints.sort_values(['Points'])
#%%
monacopoints_plt = sns.barplot(x=df_monacoPoints['Points'], y=df_monacoPoints['Driver'], orient='h', color='#FF1801')
monacopoints_plt.tick_params(labelsize=12, colors='white')
monacopoints_plt.set_title("Points earned at Monaco Grand Prix by Driver", fontsize=22, color='white')
monacopoints_plt.set_xlabel("Points", fontsize=19, color='white')
monacopoints_plt.set_ylabel("Driver", fontsize=19, color='white')
#%% Most dominant driver in Azerbaijan GP

# Purely wins
df_azerbaijanWins = pd.DataFrame(df_azerbaijan['1'].value_counts().reset_index())

# Total points scored, (1st Place : 25 points, 2nd Place : 18, 3rd Place : 15})
df_azerbaijanPlacements = pd.DataFrame(df_azerbaijan.iloc[:, 2:12])

# Purely points scored
azerbaijan_drivers = []
for c in df_azerbaijanPlacements.columns:
    counts = df_azerbaijanPlacements[c].value_counts()
    for name, points in counts.items():
        if name not in azerbaijan_drivers:
            azerbaijan_drivers.append(name)


'''   
df_drivers = pd.DataFrame(columns=sing_drivers)
df_drivers.append(pd.Series(), ignore_index=True)
df1 = pd.DataFrame([[0] * len(df_drivers.columns)], columns=df_drivers.columns)
df_drivers = df1.append(df_drivers, ignore_index=True)
'''
df_azerbaijan_drivers = pd.DataFrame(columns = ['Driver', 'Points'])
df_azerbaijan_drivers = df_azerbaijan_drivers.append({'Driver':azerbaijan_drivers}, ignore_index=True)
df_azerbaijan_drivers = df_azerbaijan_drivers.explode('Driver')
df_azerbaijan_drivers = df_azerbaijan_drivers.set_index('Driver')

df_azerbaijanPoints = df_azerbaijan_drivers.copy()
df_azerbaijanPoints['Points'] = 0

"""
for c in df_singaporePlacements.columns:
    counts = df_singaporePlacements[c].value_counts()
    for name, points in counts.items():
        sing_points.at[name] = sing_points.at[name] + (counts[name] * placements[int(c)-1])
"""

for c in df_azerbaijanPlacements.columns:
    counts = df_azerbaijanPlacements[c].value_counts()
    for name, points in counts.items():
        df_azerbaijanPoints.at[name, 'Points'] += (counts[name] * placements[int(c)-1])

df_azerbaijanPoints = df_azerbaijanPoints.reset_index()
df_azerbaijanPoints = df_azerbaijanPoints.sort_values(['Points'])
#%%
azerbaijanpoints_plt = sns.barplot(x=df_azerbaijanPoints['Points'], y=df_azerbaijanPoints['Driver'], orient='h', color='red')
azerbaijanpoints_plt.tick_params(labelsize=12, colors='white')
azerbaijanpoints_plt.set_title("Points earned at Azerbaijan Grand Prix by Driver", fontsize=22, color='white')
azerbaijanpoints_plt.set_xlabel("Points", fontsize=19, color='white')
azerbaijanpoints_plt.set_ylabel("Driver", fontsize=19, color='white')
#%% Most dominant driver in Japan GP

# Purely wins
df_japanWins = pd.DataFrame(df_japan['1'].value_counts().reset_index())

# Total points scored, (1st Place : 25 points, 2nd Place : 18, 3rd Place : 15})
df_japanPlacements = pd.DataFrame(df_japan.iloc[:, 2:12])

# Purely points scored
japan_drivers = []
for c in df_japanPlacements.columns:
    counts = df_japanPlacements[c].value_counts()
    for name, points in counts.items():
        if name not in japan_drivers:
            japan_drivers.append(name)


'''   
df_drivers = pd.DataFrame(columns=sing_drivers)
df_drivers.append(pd.Series(), ignore_index=True)
df1 = pd.DataFrame([[0] * len(df_drivers.columns)], columns=df_drivers.columns)
df_drivers = df1.append(df_drivers, ignore_index=True)
'''
df_japan_drivers = pd.DataFrame(columns = ['Driver', 'Points'])
df_japan_drivers = df_japan_drivers.append({'Driver':japan_drivers}, ignore_index=True)
df_japan_drivers = df_japan_drivers.explode('Driver')
df_japan_drivers = df_japan_drivers.set_index('Driver')

df_japanPoints = df_japan_drivers.copy()
df_japanPoints['Points'] = 0

"""
for c in df_singaporePlacements.columns:
    counts = df_singaporePlacements[c].value_counts()
    for name, points in counts.items():
        sing_points.at[name] = sing_points.at[name] + (counts[name] * placements[int(c)-1])
"""

for c in df_japanPlacements.columns:
    counts = df_japanPlacements[c].value_counts()
    for name, points in counts.items():
        df_japanPoints.at[name, 'Points'] += (counts[name] * placements[int(c)-1])

df_japanPoints = df_japanPoints.reset_index()
df_japanPoints = df_japanPoints[df_japanPoints['Points'] >= 65]
df_japanPoints = df_japanPoints.sort_values(['Points'])
#%%
japanpoints_plt = sns.barplot(x=df_japanPoints['Points'], y=df_japanPoints['Driver'], orient='h', color='red')
japanpoints_plt.tick_params(labelsize=12, colors='white')
japanpoints_plt.set_title("Points earned at Japan Grand Prix by Driver", fontsize=22, color='white')
japanpoints_plt.set_xlabel("Points", fontsize=19, color='white')
japanpoints_plt.set_ylabel("Driver", fontsize=19, color='white')


#%% Points Scored by Year

path4 = r"C:\Users\jackw\Downloads\archive (2)\DriverStandingsCleaned.xlsx"
df_driver_points_adjusted = pd.DataFrame(pd.read_excel(path4))

df_yearlyPoints = df_driver_points_adjusted[df_driver_points_adjusted['Pos'] == 1]

yearlyPoints_plt = sns.lineplot(x='Year', y='PTS', data=df_yearlyPoints, marker='.')
yearlyPoints_plt.tick_params(labelsize=12, colors='white')
yearlyPoints_plt.set_title("Points Earned by Winning Driver by Year (1950-2020)", fontsize=22, color='white')
yearlyPoints_plt.set_xlabel("Points", fontsize=19, color='white')
yearlyPoints_plt.set_ylabel("Year", fontsize=19, color='white')

#%% Places Heatmap
df_places = df_races['1'].value_counts().to_frame()

for i in range(1,5):
    df_places.insert(i, str(i+1), 0)
    i+=1

second = df_races['2'].value_counts()
for name, count in second.items():
    df_places.at[name, '2'] = count

third = df_races['3'].value_counts()
for name, count in third.items():
    df_places.at[name, '3'] = count

fourth = df_races['4'].value_counts()
for name, count in fourth.items():
    df_places.at[name, '4'] = count

fifth =  df_races['5'].value_counts()
for name, count in fifth.items():
    df_places.at[name, '5'] = count

df_places= df_places.reset_index()
df_places = df_places.truncate(after=12)
#%%

df_places= df_places.set_index('index')
#%%
places_heatmap = sns.heatmap(df_places, annot=True, cbar=False)
places_heatmap.tick_params(labelsize=12, colors='white')
places_heatmap.set_title("Positions Finished by Top 13 Drivers", fontsize=22, color='white')
places_heatmap.set_xlabel("Place", fontsize=19, color='white')
places_heatmap.set_ylabel("Driver", fontsize=19, color='white')
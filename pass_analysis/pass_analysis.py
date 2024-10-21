# Loading the teams data
import pandas as pd
import json
from mplsoccer import Pitch, Sbopen, VerticalPitch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


path_teams='data/teams.json'
path_matches='data/matches_World_Cup.json'
path_events_world_cup = "data/events_World_Cup.json"
path_players = 'data/players.json'
path_tags = 'data/tags2name.csv'
path_events_name = 'data/eventid2name.csv'



# Data loading optimization
def load_json(path):
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

df_teams = load_json(path_teams)
df_matches = load_json(path_matches)
world_cup_events = load_json(path_events_world_cup)
players_world_cup = load_json(path_players)


# Selecting the matches of the Polish team. Their team ID is 13869
polish_matches = df_matches[df_matches['teamsData'].apply(lambda x: '13869' in x.keys())]
polish_matches_list = list(polish_matches.wyId)


#structure of data
polish_matches_events = world_cup_events[world_cup_events.matchId.isin(polish_matches_list)]
#polish_matches_events.head()

polish_players_wc = players_world_cup[players_world_cup['passportArea'].apply(lambda x: x['name'])=='Poland']

polish_players = polish_players_wc[['shortName','wyId','foot','lastName']].rename(columns={'wyId':'playerId'})

tags = pd.read_csv(path_tags)
events_name = pd.read_csv(path_events_name)
#tags.head()

#Data Cleaning



#matches_events
# Create a copy of the DataFrame before modifying it
polish_matches_events = polish_matches_events.copy()

# Use .loc to set values
polish_matches_events.loc[:, ['y', 'x', 'end_y', 'end_x']] = polish_matches_events['positions'].apply(lambda x: pd.Series({'y': x[0]['y'], 'x': x[0]['x'], 'end_y': x[1]['y'], 'end_x': x[1]['x']}))

#players_info

# extract players info
players_info = []
for match_id, team_data in zip(polish_matches['wyId'], polish_matches['teamsData']):
    for team_id, team_info in team_data.items():
        for player_info in team_info['formation']['lineup']:
            player_info['teamId'] = team_id
            player_info['inFormation'] = True
            player_info['sub'] = False
            player_info['minute'] = None
            player_info['playerIn'] = ''
            player_info['playerOut'] = ''
            player_info['matchId'] = match_id
            players_info.append(player_info)
        for player_info in team_info['formation']['bench']:
            player_info['teamId'] = team_id
            player_info['inFormation'] = False
            player_info['sub'] = False
            player_info['minute'] = None
            player_info['playerIn'] = ''
            player_info['playerOut'] = ''
            player_info['matchId'] = match_id
            players_info.append(player_info)
        for sub_info in team_info['formation']['substitutions']:
            player_in = [p for p in players_info if p['playerId'] == sub_info['playerIn']][0]
            player_out = [p for p in players_info if p['playerId'] == sub_info['playerOut']][0]
            player_out['sub'] = True
            player_out['minute'] = sub_info['minute']
            player_out['playerIn'] = sub_info['playerIn']
            player_out['playerOut'] = sub_info['playerOut']
            player_out['matchId'] = match_id
            player_in['inFormation'] = False
            player_in['sub'] = True
            player_in['minute'] = sub_info['minute']
            player_in['playerIn'] = sub_info['playerIn']
            player_in['playerOut'] = sub_info['playerOut']
            player_in['matchId'] = match_id

# create a DataFrame from the players info
players_df = pd.DataFrame(players_info)

# select the desired columns
players_df = players_df[['matchId', 'teamId', 'playerId', 'inFormation', 'sub', 'minute', 'playerIn', 'playerOut']]

# print the DataFrame
polish_players = pd.merge(players_df, polish_players, on='playerId', how='left')

# filter only players for polish team
polish_players = polish_players[polish_players.teamId == '13869']

def decode_name(name):
    if isinstance(name, str):
        return name.encode('latin1').decode('unicode_escape')
    return name

polish_players['lastName'] = polish_players['lastName'].apply(decode_name)

#matches_info

polish_matches = polish_matches[['wyId', 'label']].rename(columns={"wyId": "matchId"})

polish_matches_events = polish_matches_events.merge(polish_matches, on='matchId', how='left')

# Convert the 'tags' column to a list of tag IDs
polish_matches_events['tags'] = polish_matches_events['tags'].apply(lambda tag_list: [tag['id'] for tag in tag_list])

# One-hot encode the tag IDs using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
tags_dummies = pd.DataFrame(mlb.fit_transform(polish_matches_events['tags']), columns=mlb.classes_, index=polish_matches_events.index)

# Create a mapping of tag IDs to labels
id_to_label = tags.set_index('Tag')['Label'].to_dict()

# Rename the columns of tags_dummies using the id_to_label mapping
tags_dummies.columns = tags_dummies.columns.map(id_to_label)

# Merge the one-hot encoded DataFrame with the original polish_matches_events dataset
polish_matches_events = pd.concat([polish_matches_events, tags_dummies], axis=1)

# Drop the original 'tags' column
polish_matches_events = polish_matches_events.drop(columns=['tags'])

#polish_matches_events.head()


polish_players_names = polish_players.loc[:, ['playerId', 'lastName']].drop_duplicates(subset=['playerId', 'lastName'])
#polish_players_names.head()

df_polish_mathes_events = polish_matches_events[(polish_matches_events.eventName=='Shot') & (polish_matches_events.teamId == 13869)].rename(columns={'label': 'matchName'})
df_polish_mathes_events = pd.merge(df_polish_mathes_events, polish_players_names, on='playerId')


# Define a custom function to determine the shot outcome
def shot_outcome(row):
    if row['Goal'] == 1:
        return 'Goal'
    elif row['accurate'] == 1:
        return 'Accurate'
    elif row['not accurate'] == 1:
        return 'Missed'

# Apply the custom function to create the 'shot_outcome' column
df_polish_mathes_events['shot_outcome'] = df_polish_mathes_events.apply(shot_outcome, axis=1)

#df_polish_mathes_events.head()




plot_df = df_polish_mathes_events[(df_polish_mathes_events.matchId == 2057996)].copy()
pitch = Pitch(line_zorder=2, line_color="black")
fig, ax = pitch.draw(figsize=(12, 8))
# Size of the pitch in yards
pitchLengthX = 120
pitchWidthY = 80
# Standardize the 'x' and 'y' values
plot_df['x_standardized'] = plot_df['x'] / 100 * pitchLengthX
plot_df['y_standardized'] = plot_df['y'] / 100 * pitchWidthY
# Plot the shots by looping through them.
for i, shot in plot_df.iterrows():
    # Get the information
    x = shot['x_standardized']
    y = shot['y_standardized']
    # Set circle size
    circleSize = 2

    # Set color based on shot outcome
    if shot.shot_outcome == 'Goal':
        pitch.scatter(x, y, alpha=1, s=500, color='green', ax=ax)
        pitch.annotate(shot["lastName"], (x + 1, y - 2), ax=ax, fontsize=12)
    elif shot.shot_outcome == 'Accurate':
        pitch.scatter(x, y, alpha=1, s=500, color='blue', ax=ax)
        pitch.annotate(shot["lastName"], (x + 1, y - 2), ax=ax, fontsize=12)
    elif shot.shot_outcome == 'Missed':
        pitch.scatter(x, y, alpha=1, s=500, color='red', ax=ax)
        pitch.annotate(shot["lastName"], (x + 1, y - 2), ax=ax, fontsize=12)

# Create legend
goal_legend = plt.Line2D([0], [0], color="green", lw=4)
accurate_legend = plt.Line2D([0], [0], color="blue", lw=4)
missed_legend = plt.Line2D([0], [0], color="red", lw=4)
ax.legend([goal_legend, accurate_legend, missed_legend], ['Goal', 'Accurate', 'Missed'], loc='upper right')

# Set title
match_name = plot_df['matchName'].iloc[0]  # Get the match name from the first row of plot_df
fig.suptitle(match_name, fontsize=24)
fig.set_size_inches(12, 8)
#plt.show()
file_name = '{}_plot.jpg'.format(match_name)
plt.savefig("figures/"+file_name)
plt.close()

df_polish_mathes_events = polish_matches_events[(polish_matches_events.eventName=='Pass') & (polish_matches_events.teamId == 13869)].rename(columns={'label': 'matchName'})
df_polish_mathes_events = pd.merge(df_polish_mathes_events, polish_players_names, on='playerId')
#df_polish_mathes_events.head()

#prepare the dataframe of passes by England that were no-throw ins
mask_poland = (df_polish_mathes_events.subEventName != "Throw-in") & (df_polish_mathes_events.matchId == 2057996)
df_passes = df_polish_mathes_events.loc[mask_poland, ['x', 'y', 'end_x', 'end_y', 'lastName', 'accurate']]
#get the list of all players who made a pass
names = df_passes['lastName'].unique()

#draw 4x4 pitches
pitchLengthX = 120
pitchWidthY = 80
pitch = Pitch(line_color='black', pad_top=20, pitch_length=pitchLengthX, pitch_width=pitchWidthY)

fig, axs = pitch.grid(ncols=4, nrows=4, grid_height=0.85, title_height=0.06, axis=False,
                      endnote_height=0.04, title_space=0.04, endnote_space=0.01)

#standarize x and y
df_passes['x'] = df_passes['x'] / 100 * pitchLengthX
df_passes['y'] = df_passes['y'] / 100 * pitchWidthY
df_passes['end_x'] = df_passes['end_x'] / 100 * pitchLengthX
df_passes['end_y'] = df_passes['end_y'] / 100 * pitchWidthY

#for each player
for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
    player_df = df_passes.loc[df_passes["lastName"] == name]

    # Calculate the share of accurate passes
    total_passes = len(player_df)
    accurate_passes = len(player_df[player_df['accurate'] == 1])
    accuracy_share = accurate_passes / total_passes * 100
    
    # Put player name and accuracy share over the plot
    ax.text(60, -10, f"{name} ({accuracy_share:.1f}%)", ha='center', va='center', fontsize=14)
    
    # Plot arrow and scatter
    for idx, row in player_df.iterrows():
        arrow_color = "green" if row['accurate'] == 1 else "red"
        pitch.arrows(row.x, row.y,
                     row.end_x, row.end_y, color=arrow_color, ax=ax, width=1)
        pitch.scatter(row.x, row.y, alpha=0.2, s=50, color=arrow_color, ax=ax)

#We have more than enough pitches - remove them
for ax in axs['pitch'][-1, 16 - len(names):]:
    ax.remove()

#Another way to set title using mplsoccer
axs['title'].text(0.5, 0.5, 'Polish passes against Senegal', ha='center', va='center', fontsize=30)
#plt.show()
plt.savefig("figures/"+'Polish_passes.jpg')
plt.close()

#prepare the dataframe of passes by England that were no-throw ins
mask_poland = (df_polish_mathes_events.subEventName != "Throw-in") & (df_polish_mathes_events.matchId == 2057996)
df_passes = df_polish_mathes_events.loc[mask_poland, ['x', 'y', 'end_x', 'end_y', 'lastName', 'accurate']]
df_passes = df_passes[df_passes.end_x > df_passes.x]

#get the list of all players who made a pass
names = df_passes['lastName'].unique()

#draw 4x4 pitches
pitchLengthX = 120
pitchWidthY = 80
pitch = Pitch(line_color='black', pad_top=20, pitch_length=pitchLengthX, pitch_width=pitchWidthY)

fig, axs = pitch.grid(ncols=4, nrows=4, grid_height=0.85, title_height=0.06, axis=False,
                      endnote_height=0.04, title_space=0.04, endnote_space=0.01)

#standarize x and y
df_passes['x'] = df_passes['x'] / 100 * pitchLengthX
df_passes['y'] = df_passes['y'] / 100 * pitchWidthY
df_passes['end_x'] = df_passes['end_x'] / 100 * pitchLengthX
df_passes['end_y'] = df_passes['end_y'] / 100 * pitchWidthY

#for each player
for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
    player_df = df_passes.loc[df_passes["lastName"] == name]

    # Calculate the share of accurate passes
    total_passes = len(player_df)
    accurate_passes = len(player_df[player_df['accurate'] == 1])
    accuracy_share = accurate_passes / total_passes * 100
    
    # Put player name and accuracy share over the plot
    ax.text(60, -10, f"{name} ({accuracy_share:.1f}%)", ha='center', va='center', fontsize=14)
    
    # Plot arrow and scatter
    for idx, row in player_df.iterrows():
        arrow_color = "green" if row['accurate'] == 1 else "red"
        pitch.arrows(row.x, row.y,
                     row.end_x, row.end_y, color=arrow_color, ax=ax, width=1)
        pitch.scatter(row.x, row.y, alpha=0.2, s=50, color=arrow_color, ax=ax)

#We have more than enough pitches - remove them
for ax in axs['pitch'][-1, 16 - len(names):]:
    ax.remove()

#Another way to set title using mplsoccer
axs['title'].text(0.5, 0.5, 'Polish offensive passes against Senegal', ha='center', va='center', fontsize=30)
#plt.show()
plt.savefig("figures/"+'Polish_offensive_passes.jpg')
plt.close()

df = polish_matches_events[(polish_matches_events.matchId == 2057996) & (polish_matches_events.teamId == 13869) & (polish_matches_events.eventName == 'Pass') & (polish_matches_events.matchPeriod == '1H')].rename(columns={'label': 'matchName'})
df['next_player'] = df['playerId'].shift(-1)
df.loc[df['accurate'] != 1, 'next_player'] = None
df = df[df['next_player'].notnull()]
df = df[['playerId','x', 'y', 'end_x', 'end_y', 'eventSec', 'next_player']]

#df.head()

polish_sen = polish_players[(polish_players.matchId == 2057996) & (polish_players.teamId == '13869')]
polish_sen = polish_sen[['playerId', 'minute', 'lastName']]
players = polish_sen[['playerId', 'lastName']].drop_duplicates()
#players.head()

merged_df = df.merge(players, left_on='playerId', right_on='playerId', how='left')
merged_df = merged_df.merge(players, left_on='next_player', right_on='playerId', how='left', suffixes=('', '_next'))
merged_df.drop('playerId_next', axis=1, inplace=True)
merged_df.drop(['playerId', 'next_player', 'eventSec'], axis=1, inplace=True)
#merged_df.head()


df_pass = merged_df.copy()

pitchLengthX = 120
pitchWidthY = 80


df_pass['x'] = df_pass['x'] / 100 * pitchLengthX
df_pass['y'] = df_pass['y'] / 100 * pitchWidthY
df_pass['end_x'] = df_pass['end_x'] / 100 * pitchLengthX
df_pass['end_y'] = df_pass['end_y'] / 100 * pitchWidthY

scatter_df = pd.DataFrame()
for i, name in enumerate(df_pass["lastName"].unique()):
    passx = df_pass.loc[df_pass["lastName"] == name]["x"].to_numpy()
    recx = df_pass.loc[df_pass["lastName_next"] == name]["end_x"].to_numpy()
    passy = df_pass.loc[df_pass["lastName"] == name]["y"].to_numpy()
    recy = df_pass.loc[df_pass["lastName_next"] == name]["end_y"].to_numpy()
    scatter_df.at[i, "lastName"] = name
    # Make sure that x and y location for each circle representing the player is the average of passes and receptions
    scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
    scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
    # Calculate number of passes
    scatter_df.at[i, "no"] = df_pass.loc[df_pass["lastName"] == name].count().iloc[0]

# Adjust the size of a circle so that the player who made more passes
scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)
#scatter_df.head()

# Counting passes between players
df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["lastName"], x["lastName_next"]])), axis=1)
lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)

# Drawing pitch

pitchLengthX = 120
pitchWidthY = 80
pitch = Pitch(line_color='grey', pitch_length=pitchLengthX, pitch_width=pitchWidthY)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
# Scatter the location on the pitch
pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder=3)
# Annotating player name
for i, row in scatter_df.iterrows():
    pitch.annotate(row.lastName, xy=(row.x, row.y), c='black', va='center', ha='center', weight="bold", size=16, ax=ax["pitch"], zorder=4)

fig.suptitle("Nodes location - Poland", fontsize=30)
#plt.show()
plt.close()

# Plot once again pitch and vertices
pitch = Pitch(line_color='grey')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder=3)
for i, row in scatter_df.iterrows():
    pitch.annotate(row.lastName, xy=(row.x, row.y), c='black', va='center', ha='center', weight="bold", size=16, ax=ax["pitch"], zorder=4)

for i, row in lines_df.iterrows():
    player1 = row["pair_key"].split("_")[0]
    player2 = row['pair_key'].split("_")[1]
    # Take the average location of players to plot a line between them
    player1_x = scatter_df.loc[scatter_df["lastName"] == player1]['x'].iloc[0]
    player1_y = scatter_df.loc[scatter_df["lastName"] == player1]['y'].iloc[0]
    player2_x = scatter_df.loc[scatter_df["lastName"] == player2]['x'].iloc[0]
    player2_y = scatter_df.loc[scatter_df["lastName"] == player2]['y'].iloc[0]
    num_passes = row["pass_count"]
    # Adjust the line width so that the more passes, the wider the line
    line_width = (num_passes / lines_df['pass_count'].max() * 10)
    # Plot lines on the pitch
    pitch.lines(player1_x, player1_y, player2_x, player2_y,
                alpha=1, lw=line_width, zorder=2, color="red", ax=ax["pitch"])

fig.suptitle("Poland Passing Network against Senegal", fontsize=30)
#plt.show()
plt.savefig("figures/"+'Polish_Passing_Network.jpg')
plt.close()

#calculate number of successful passes by player
no_passes = df_pass.groupby(['lastName']).x.count().reset_index()
no_passes.rename({'x':'pass_count'}, axis='columns', inplace=True)
#find one who made most passes
max_no = no_passes["pass_count"].max()
#calculate the denominator - 10*the total sum of passes
denominator = 10*no_passes["pass_count"].sum()
#calculate the nominator
nominator = (max_no - no_passes["pass_count"]).sum()
#calculate the centralisation index
centralisation_index = nominator/denominator
print("Centralisation index is ", centralisation_index)

df_pass = merged_df.copy()

pitchLengthX = 120
pitchWidthY = 80


df_pass['x'] = df_pass['x'] / 100 * pitchLengthX
df_pass['y'] = df_pass['y'] / 100 * pitchWidthY
df_pass['end_x'] = df_pass['end_x'] / 100 * pitchLengthX
df_pass['end_y'] = df_pass['end_y'] / 100 * pitchWidthY

df_pass = df_pass[df_pass.end_x > df_pass.x]

scatter_df = pd.DataFrame()
for i, name in enumerate(df_pass["lastName"].unique()):
    passx = df_pass.loc[df_pass["lastName"] == name]["x"].to_numpy()
    recx = df_pass.loc[df_pass["lastName_next"] == name]["end_x"].to_numpy()
    passy = df_pass.loc[df_pass["lastName"] == name]["y"].to_numpy()
    recy = df_pass.loc[df_pass["lastName_next"] == name]["end_y"].to_numpy()
    scatter_df.at[i, "lastName"] = name
    # Make sure that x and y location for each circle representing the player is the average of passes and receptions
    scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
    scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
    # Calculate number of passes
    scatter_df.at[i, "no"] = df_pass.loc[df_pass["lastName"] == name].count().iloc[0]

# Adjust the size of a circle so that the player who made more passes
scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)

# Counting passes between players
df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["lastName"], x["lastName_next"]])), axis=1)
lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)


# Drawing pitch

pitchLengthX = 120
pitchWidthY = 80
# pitch = Pitch(line_color='grey', pitch_length=pitchLengthX, pitch_width=pitchWidthY)
# fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                      endnote_height=0.04, title_space=0, endnote_space=0)
# # Scatter the location on the pitch
# pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder=3)
# Annotating player name
for i, row in scatter_df.iterrows():
    pitch.annotate(row.lastName, xy=(row.x, row.y), c='black', va='center', ha='center', weight="bold", size=16, ax=ax["pitch"], zorder=4)

# Plot once again pitch and vertices
pitch = Pitch(line_color='grey')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder=3)

for i, row in scatter_df.iterrows():
    pitch.annotate(row.lastName, xy=(row.x, row.y), c='black', va='center', ha='center', weight="bold", size=16, ax=ax["pitch"], zorder=4)

for i, row in lines_df.iterrows():
    player1 = row["pair_key"].split("_")[0]
    player2 = row['pair_key'].split("_")[1]
    # Take the average location of players to plot a line between them
    player1_x = scatter_df.loc[scatter_df["lastName"] == player1]['x'].iloc[0]
    player1_y = scatter_df.loc[scatter_df["lastName"] == player1]['y'].iloc[0]
    player2_x = scatter_df.loc[scatter_df["lastName"] == player2]['x'].iloc[0]
    player2_y = scatter_df.loc[scatter_df["lastName"] == player2]['y'].iloc[0]
    num_passes = row["pass_count"]
    # Adjust the line width so that the more passes, the wider the line
    line_width = (num_passes / lines_df['pass_count'].max() * 10)
    # Plot lines on the pitch
    pitch.lines(player1_x, player1_y, player2_x, player2_y,
                alpha=1, lw=line_width, zorder=2, color="red", ax=ax["pitch"])

fig.suptitle("Poland Offensive Passing Network against Senegal", fontsize=30)
#plt.show()
plt.savefig("figures/"+'Polish_Offensive_Passing_Network.jpg')
plt.close()

df_polish_mathes_events = polish_matches_events[(polish_matches_events.eventName=='Pass') & (polish_matches_events.teamId == 13869)].rename(columns={'label': 'matchName'})
df_polish_mathes_events = pd.merge(df_polish_mathes_events, polish_players_names, on='playerId')
df_polish_mathes_events = df_polish_mathes_events[['lastName', 'accurate']]
# group by lastName and compute count and sum
df_polish_mathes_events = df_polish_mathes_events.groupby('lastName').agg({'accurate': ['count', 'sum']})

# rename the columns
df_polish_mathes_events.columns = ['count', 'sum']

# reset the index
df_polish_mathes_events = df_polish_mathes_events.reset_index()
df_polish_mathes_events['sucPass'] = df_polish_mathes_events['sum'] / df_polish_mathes_events['count']

# set the plot size
fig, ax = plt.subplots(figsize=(10, 8))

# create the scatter plot
df_polish_mathes_events.plot(kind='scatter', x='count', y='sucPass', color='blue', ax=ax)

# set the axis labels and title
ax.set_xlabel('Count')
ax.set_ylabel('Success Pass %')
ax.set_title('Scatter Plot of Count vs Success Pass %')

# add labels to the scatter plot
for i, row in df_polish_mathes_events.iterrows():
    ax.text(row['count'], row['sucPass']+0.005, row['lastName'], ha='center', va='center')

# display the plot
#plt.show()
plt.savefig("figures/"+'Passing_Scatterplot.jpg')
plt.close()


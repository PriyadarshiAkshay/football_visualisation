# Importing the necessary libraries
import pandas as pd
import json
from mplsoccer import Pitch, Sbopen, VerticalPitch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Data loading from json files
def load_json(path):
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

# Get the wyId of the team
def get_team_wyId(df_teams,team_name):
    team_wyId = df_teams.loc[df_teams['name'] == team_name, 'wyId'].values
    return int(team_wyId[0]) if len(team_wyId) > 0 else None

# Select the opponent
def get_opponent_name(names_of_opponents):
    while True:
        input_opponent = input("Enter the name of the opponent: ")
        if input_opponent in names_of_opponents:
            return input_opponent
        print("Invalid input. Please enter a valid opponent name from:", names_of_opponents)

def decode_name(name):
    return name.encode('utf-8').decode('unicode_escape') if isinstance(name, str) else name

# Define a custom function to determine the shot outcome
def shot_outcome(row):
    if row['Goal'] == 1:
        return 'Goal'
    elif row['accurate'] == 1:
        return 'Accurate'
    elif row['not accurate'] == 1:
        return 'Missed'

class PassAnalysis:
    def __init__(self, team_name, opponent_name=None, offensive_passes=True, include_passes=True, figures_folder="figures/"):
        self.team_name = team_name
        self.opponent_name = opponent_name
        self.offensive_passes = offensive_passes
        self.include_passes = include_passes
        self.figures_folder = figures_folder

        if self.figures_folder[-1] != "/":
            self.figures_folder += "/"

        if not os.path.exists(self.figures_folder):
            os.makedirs(self.figures_folder)
        print(f"Figures will be saved in {self.figures_folder}")
    


    def load_and_analyse_data(self,):

        # Setting the paths to the data files
        path_teams = 'data/teams.json'
        path_matches = 'data/matches_World_Cup.json'
        path_events_world_cup = "data/events_World_Cup.json"
        path_players = 'data/players.json'
        path_tags = 'data/tags2name.csv'
        path_events_name = 'data/eventid2name.csv'

        # Load the data
        df_teams = load_json(path_teams)
        df_matches = load_json(path_matches)
        world_cup_events = load_json(path_events_world_cup)
        players_world_cup = load_json(path_players)
        tags = pd.read_csv(path_tags)
        events_name = pd.read_csv(path_events_name)

        # Get the wyId of the selected team
        self.team_wyId = get_team_wyId(df_teams,self.team_name)
        #print(f"The wyId for {team_name} is {team_wyId}")

        # Selecting the matches of the team.
        team_matches = df_matches[df_matches['teamsData'].apply(lambda x: str(self.team_wyId) in x.keys())]
        team_matches_list = list(team_matches.wyId)

        ## Get the list of opponents
        list_of_opponents = [item for sublist in team_matches['teamsData'].apply(lambda x: list(x.keys())) for item in sublist]
        list_of_opponents = [team_id for team_id in list_of_opponents if team_id != str(self.team_wyId)]
        names_of_opponents=[df_teams[df_teams['wyId']==int(x)]['name'].values[0] for x in list_of_opponents]

        # Get the opponent name from the user if not provided
        if self.opponent_name is None:
            print(f"The opponents were: \n{names_of_opponents}")
            self.opponent_name = get_opponent_name(names_of_opponents)

        # Get the wyId of the selected opponent
        selected_opponent_id=list_of_opponents[names_of_opponents.index(self.opponent_name)]

        # Select the match
        filtered_matches = team_matches[team_matches['teamsData'].apply(lambda x: selected_opponent_id in x.keys())]
        filtered_matches_list=list(filtered_matches.wyId)
        self.selected_match=int(filtered_matches['wyId'].iloc[0])

        #Select the events of the selected match
        team_matches_events = world_cup_events[world_cup_events.matchId.isin([self.selected_match])]
        team_matches_events.loc[:, ['y', 'x', 'end_y', 'end_x']] = team_matches_events['positions'].apply(lambda x: pd.Series({'y': x[0]['y'], 'x': x[0]['x'], 'end_y': x[1]['y'], 'end_x': x[1]['x']}))


        players_world_cup = players_world_cup[['shortName','wyId','foot','lastName','firstName']].rename(columns={'wyId':'playerId'})


        # extract players info
        players_info = []
        for match_id, team_data in zip(filtered_matches['wyId'], filtered_matches['teamsData']):
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
        match_players = pd.merge(players_df, players_world_cup, on='playerId', how='left')
        team1_players = match_players[match_players.teamId == str(self.team_wyId)]

        team1_players.loc[:, 'lastName'] = team1_players['lastName'].apply(decode_name)
        team1_players.loc[:, 'firstName'] = team1_players['firstName'].apply(decode_name)
        team1_players.loc[:, 'shortName'] = team1_players['shortName'].apply(decode_name)

        filtered_matches = filtered_matches[['wyId', 'label']].rename(columns={"wyId": "matchId"})
        team_matches_events = team_matches_events.merge(filtered_matches, on='matchId', how='left')

        # Convert the 'tags' column to a list of tag IDs
        team_matches_events['tags'] = team_matches_events['tags'].apply(lambda tag_list: [tag['id'] for tag in tag_list])

        # One-hot encode the tag IDs using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        tags_dummies = pd.DataFrame(mlb.fit_transform(team_matches_events['tags']), columns=mlb.classes_, index=team_matches_events.index)

        # Create a mapping of tag IDs to labels
        id_to_label = tags.set_index('Tag')['Label'].to_dict()

        # Rename the columns of tags_dummies using the id_to_label mapping
        tags_dummies.columns = tags_dummies.columns.map(id_to_label)

        # Merge the one-hot encoded DataFrame with the original team_matches_events dataset
        team_matches_events = pd.concat([team_matches_events, tags_dummies], axis=1)

        # Drop the original 'tags' column
        self.team_matches_events = team_matches_events.drop(columns=['tags'])

        self.team1_players_names = team1_players.loc[:, ['playerId', 'lastName', 'firstName']].drop_duplicates(subset=['playerId', 'lastName','firstName'])

        self.team1_players = team1_players

    ##### common stuff ####

    def plot_shots(self,):

        df_team1_matches_events = self.team_matches_events[(self.team_matches_events.eventName=='Shot') & (self.team_matches_events.teamId == self.team_wyId)].rename(columns={'label': 'matchName'})
        df_team1_matches_events = pd.merge(df_team1_matches_events, self.team1_players_names, on='playerId')

        # Apply the custom function to create the 'shot_outcome' column
        df_team1_matches_events['shot_outcome'] = df_team1_matches_events.apply(shot_outcome, axis=1)
        plot_df = df_team1_matches_events[(df_team1_matches_events.matchId == self.selected_match)].copy()

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
        ax.legend([goal_legend, accurate_legend, missed_legend], ['Goal', 'Accurate', 'Missed'], loc='upper right',bbox_to_anchor=(.97, .96),fontsize=12)

        # Set title
        match_name = plot_df['matchName'].iloc[0]  # Get the match name from the first row of plot_df
        fig.suptitle(match_name, fontsize=24)
        fig.set_size_inches(12, 8)
        #plt.show()
        file_name = f"shots_{self.team_name}_{self.opponent_name}_plot.png"
        plt.savefig(self.figures_folder+file_name, bbox_inches='tight',dpi=300, edgecolor='none')
        plt.close()

    def plot_passes(self,):

        df_team1_matches_events = self.team_matches_events[(self.team_matches_events.eventName=='Pass') & (self.team_matches_events.teamId == self.team_wyId)].rename(columns={'label': 'matchName'})
        df_team1_matches_events = pd.merge(df_team1_matches_events, self.team1_players_names, on='playerId')

        #prepare the dataframe of passes by team1 that were no-throw ins
        mask_team1 = (df_team1_matches_events.subEventName != "Throw-in") & (df_team1_matches_events.matchId == self.selected_match)
        df_passes = df_team1_matches_events.loc[mask_team1, ['x', 'y', 'end_x', 'end_y', 'lastName', 'accurate']]

        fname = 'all'
        if self.offensive_passes:
            df_passes = df_passes[df_passes.end_x > df_passes.x]
            fname = 'offensive'

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
            if total_passes == 0:
                continue
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
        axs['title'].text(0.5, 0.5, f"{self.team_name} {fname} passes against {self.opponent_name}", ha='center', va='center', fontsize=30)
        #plt.show()
        plt.savefig(self.figures_folder+f"{self.team_name}_{fname}_passes_{self.opponent_name}.png", bbox_inches='tight',dpi=300, edgecolor='none')
        plt.close()

    def plot_pass_network(self,):

        df = self.team_matches_events[(self.team_matches_events.matchId == self.selected_match) & (self.team_matches_events.teamId == self.team_wyId) & (self.team_matches_events.eventName == 'Pass') & (self.team_matches_events.matchPeriod == '1H')].rename(columns={'label': 'matchName'})
        df['next_player'] = df['playerId'].shift(-1)
        df.loc[df['accurate'] != 1, 'next_player'] = None
        df = df[df['next_player'].notnull()]
        df = df[['playerId','x', 'y', 'end_x', 'end_y', 'eventSec', 'next_player']]


        team1_team2 = self.team1_players[(self.team1_players.matchId == self.selected_match) & (self.team1_players.teamId == str(self.team_wyId))]
        team1_team2 = team1_team2[['playerId', 'minute', 'lastName']]
        players = team1_team2[['playerId', 'lastName']].drop_duplicates()

        merged_df = df.merge(players, left_on='playerId', right_on='playerId', how='left')
        merged_df = merged_df.merge(players, left_on='next_player', right_on='playerId', how='left', suffixes=('', '_next'))
        merged_df.drop('playerId_next', axis=1, inplace=True)
        merged_df.drop(['playerId', 'next_player', 'eventSec'], axis=1, inplace=True)

        df_pass = merged_df.copy()

        pitchLengthX = 120
        pitchWidthY = 80


        df_pass['x'] = df_pass['x'] / 100 * pitchLengthX
        df_pass['y'] = df_pass['y'] / 100 * pitchWidthY
        df_pass['end_x'] = df_pass['end_x'] / 100 * pitchLengthX
        df_pass['end_y'] = df_pass['end_y'] / 100 * pitchWidthY

        fname = 'all'
        if self.offensive_passes:
            df_pass = df_pass[df_pass.end_x > df_pass.x]
            fname = 'offensive'

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
        pitch = Pitch(line_color='grey', pitch_length=pitchLengthX, pitch_width=pitchWidthY)
        fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                            endnote_height=0.04, title_space=0, endnote_space=0)
        # Scatter the location on the pitch
        pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder=3)
        # Annotating player name
        for i, row in scatter_df.iterrows():
            pitch.annotate(row.lastName, xy=(row.x, row.y), c='black', va='center', ha='center', weight="bold", size=16, ax=ax["pitch"], zorder=4)

        if self.include_passes:
            # Plot lines between players
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


        fig.suptitle(f"Nodes location - {self.team_name} {fname}", fontsize=30)
        #plt.show()
        plt.savefig(self.figures_folder+f"Nodes_{self.team_name}_{fname}.png", bbox_inches='tight',dpi=300, edgecolor='none')
        plt.close()
        self.df_pass = df_pass
        return df_pass

    def centralisation_index(self,df_pass):

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

    def passing_scatterplot(self,):
        df_team1_matches_events = self.team_matches_events[(self.team_matches_events.eventName=='Pass') & (self.team_matches_events.teamId == self.team_wyId)].rename(columns={'label': 'matchName'})
        df_team1_matches_events = pd.merge(df_team1_matches_events, self.team1_players_names, on='playerId')
        df_team1_matches_events = df_team1_matches_events[['lastName', 'accurate']]
        # group by lastName and compute count and sum
        df_team1_matches_events = df_team1_matches_events.groupby('lastName').agg({'accurate': ['count', 'sum']})

        # rename the columns
        df_team1_matches_events.columns = ['count', 'sum']

        # reset the index
        df_team1_matches_events = df_team1_matches_events.reset_index()
        df_team1_matches_events['sucPass'] = df_team1_matches_events['sum'] / df_team1_matches_events['count']

        # set the plot size
        fig, ax = plt.subplots(figsize=(10, 7))

        # create the scatter plot
        df_team1_matches_events.plot(kind='scatter', x='count', y='sucPass', color='blue', ax=ax)

        # set the axis labels and title
        ax.set_xlabel('Count', fontsize=14)
        ax.set_ylabel('Success Pass %', fontsize=14)
        ax.set_title('Scatter Plot of Count vs Success Pass %', fontsize=16)

        # add labels to the scatter plot
        for i, row in df_team1_matches_events.iterrows():
            ax.text(row['count'], row['sucPass']+0.01, row['lastName'], ha='center', va='center', fontsize=10)

        # display the plot
        #plt.show()
        plt.savefig(self.figures_folder+'Passing_Scatterplot.png', bbox_inches='tight',dpi=300, edgecolor='none')
        plt.close()

    def plot_passes_gif(self,):

        df_team1_matches_events = self.team_matches_events[(self.team_matches_events.eventName=='Pass') & (self.team_matches_events.teamId == self.team_wyId)].rename(columns={'label': 'matchName'})
        df_team1_matches_events = pd.merge(df_team1_matches_events, self.team1_players_names, on='playerId')

        mask_team1 = (df_team1_matches_events.subEventName != "Throw-in") & (df_team1_matches_events.matchId == self.selected_match)
        df_passes = df_team1_matches_events.loc[mask_team1, ['x', 'y', 'end_x', 'end_y', 'lastName', 'accurate', 'eventSec', 'matchPeriod','Goal']]

        e1_time = df_passes.loc[df_passes['matchPeriod'] == 'E1', 'eventSec'].max()
        e2_time = df_passes.loc[df_passes['matchPeriod'] == 'E2', 'eventSec'].max()

        e1_time = 0 if np.isnan(e1_time) else e1_time
        e2_time = 0 if np.isnan(e2_time) else e2_time

        # Define the mapping dictionary
        mapping = {
            '1H': 0,
            'E1': 45 * 60,
            '2H': 45 * 60 + e1_time,
            'E2': (45 * 60 + 45 * 60) + e1_time,
            'P': (45 * 60 + 45 * 60) + e1_time + e2_time,
        }

        # Replace the values in the 'matchPeriod' column
        df_passes['timestamp'] = df_passes['matchPeriod'].map(mapping)+df_passes['eventSec']

        if self.offensive_passes:
            df_passes = df_passes[df_passes.end_x > df_passes.x]

        pitchLengthX = 120
        pitchWidthY = 80
        pitch = Pitch(line_color='black', pad_top=20, pitch_length=pitchLengthX, pitch_width=pitchWidthY)

        df_passes['x'] = df_passes['x'] / 100 * pitchLengthX
        df_passes['y'] = df_passes['y'] / 100 * pitchWidthY
        df_passes['end_x'] = df_passes['end_x'] / 100 * pitchLengthX
        df_passes['end_y'] = df_passes['end_y'] / 100 * pitchWidthY

        # Create figure and axis
        fig, ax = pitch.draw(figsize=(7, 5))
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list("", ["yellow", "red"])

        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            pitch.draw(ax=ax)
            current_time = frame * 60  # Assuming 1 frame = 1 minute
            current_passes = df_passes[df_passes['timestamp'] <= current_time]
            
            for idx, row in current_passes.iterrows():
                alpha = 1 - (current_time - row['timestamp']) / (current_time + 1)  # Fade out older passes
                color = cmap(row['timestamp'] / df_passes['timestamp'].max())
                if row['Goal'] == 1:
                    color = 'green'
                pitch.arrows(row.x, row.y, row.end_x, row.end_y, color=color, alpha=max(0.2, alpha), ax=ax, width=2)
            
            ax.set_title(f"{self.team_name} passes against {self.opponent_name}\nTime: {frame} minutes")

        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=90, interval=200, repeat=False)

        # Save the animation as a GIF
        anim.save(self.figures_folder+f"{self.team_name}_passes_{self.opponent_name}.gif", writer='pillow', fps=5, dpi=300, savefig_kwargs={'facecolor': 'white'})
        plt.close()


############################################
if __name__ == "__main__":
    # Setting the environment variable to render the plots offscreen
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'


    # Setting the team and opponent names
    team_name = "Germany"
    opponent_name = "Sweden"#None
    offensive_passes = True
    include_passes = True
    figures_folder = "figures/"

    PA = PassAnalysis(team_name, opponent_name, offensive_passes, include_passes, figures_folder)

    PA.load_and_analyse_data()

    PA.plot_shots()

    PA.plot_passes()

    df_pass=PA.plot_pass_network()

    PA.centralisation_index(df_pass)

    PA.passing_scatterplot()

    # Usage
    PA.plot_passes_gif()

    ############################################
    ############################################
    #############################################

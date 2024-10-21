import pandas as pd
import json
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

class PassAnalysis:
    def __init__(self, country_name1):
        self.country_name1 = country_name1
        self.paths = {
            'teams': 'data/teams.json',
            'matches': 'data/matches_World_Cup.json',
            'events_world_cup': "data/events_World_Cup.json",
            'players': 'data/players.json',
            'tags': 'data/tags2name.csv',
            'events_name': 'data/eventid2name.csv'
        }
        self.load_data()
        self.process_data()

    def load_json(self, path):
        with open(path, encoding='utf-8') as f:
            return pd.DataFrame(json.load(f))

    def load_data(self):
        self.df_teams = self.load_json(self.paths['teams'])
        self.df_matches = self.load_json(self.paths['matches'])
        self.world_cup_events = self.load_json(self.paths['events_world_cup'])
        self.players_world_cup = self.load_json(self.paths['players'])
        self.tags = pd.read_csv(self.paths['tags'])
        self.events_name = pd.read_csv(self.paths['events_name'])

    def process_data(self):
        self.polish_matches = self.df_matches[self.df_matches['teamsData'].apply(lambda x: '13869' in x.keys())]
        self.polish_matches_list = list(self.polish_matches.wyId)
        self.polish_matches_events = self.world_cup_events[self.world_cup_events.matchId.isin(self.polish_matches_list)]
        self.polish_players_wc = self.players_world_cup[self.players_world_cup['passportArea'].apply(lambda x: x['name']) == self.country_name1]
        self.polish_players = self.polish_players_wc[['shortName', 'wyId', 'foot', 'lastName']].rename(columns={'wyId': 'playerId'})
        self.clean_data()

    def clean_data(self):
        self.polish_matches_events = self.polish_matches_events.copy()
        self.polish_matches_events.loc[:, ['y', 'x', 'end_y', 'end_x']] = self.polish_matches_events['positions'].apply(
            lambda x: pd.Series({'y': x[0]['y'], 'x': x[0]['x'], 'end_y': x[1]['y'], 'end_x': x[1]['x']}))
        self.extract_players_info()
        self.polish_players = pd.merge(self.players_df, self.polish_players, on='playerId', how='left')
        self.polish_players = self.polish_players[self.polish_players.teamId == '13869']
        self.polish_players['lastName'] = self.polish_players['lastName'].apply(self.decode_name)
        self.polish_matches = self.polish_matches[['wyId', 'label']].rename(columns={"wyId": "matchId"})
        self.polish_matches_events = self.polish_matches_events.merge(self.polish_matches, on='matchId', how='left')
        self.polish_matches_events['tags'] = self.polish_matches_events['tags'].apply(lambda tag_list: [tag['id'] for tag in tag_list])
        mlb = MultiLabelBinarizer()
        tags_dummies = pd.DataFrame(mlb.fit_transform(self.polish_matches_events['tags']), columns=mlb.classes_, index=self.polish_matches_events.index)
        id_to_label = self.tags.set_index('Tag')['Label'].to_dict()
        tags_dummies.columns = tags_dummies.columns.map(id_to_label)
        self.polish_matches_events = pd.concat([self.polish_matches_events, tags_dummies], axis=1)
        self.polish_matches_events = self.polish_matches_events.drop(columns=['tags'])
        self.polish_players_names = self.polish_players.loc[:, ['playerId', 'lastName']].drop_duplicates(subset=['playerId', 'lastName'])

    def extract_players_info(self):
        players_info = []
        for match_id, team_data in zip(self.polish_matches['wyId'], self.polish_matches['teamsData']):
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
        self.players_df = pd.DataFrame(players_info)
        self.players_df = self.players_df[['matchId', 'teamId', 'playerId', 'inFormation', 'sub', 'minute', 'playerIn', 'playerOut']]

    def decode_name(self, name):
        if isinstance(name, str):
            return name.encode('latin1').decode('unicode_escape')
        return name

    def plot_shots(self, match_id):
        plot_df = self.polish_matches_events[(self.polish_matches_events.eventName == 'Shot') & (self.polish_matches_events.teamId == 13869) & (self.polish_matches_events.matchId == match_id)].copy()
        pitch = Pitch(line_zorder=2, line_color="black")
        fig, ax = pitch.draw(figsize=(12, 8))
        pitchLengthX = 120
        pitchWidthY = 80
        plot_df['x_standardized'] = plot_df['x'] / 100 * pitchLengthX
        plot_df['y_standardized'] = plot_df['y'] / 100 * pitchWidthY
        for i, shot in plot_df.iterrows():
            x = shot['x_standardized']
            y = shot['y_standardized']
            circleSize = 2
            if shot.shot_outcome == 'Goal':
                pitch.scatter(x, y, alpha=1, s=500, color='green', ax=ax)
                pitch.annotate(shot["lastName"], (x + 1, y - 2), ax=ax, fontsize=12)
            elif shot.shot_outcome == 'Accurate':
                pitch.scatter(x, y, alpha=1, s=500, color='blue', ax=ax)
                pitch.annotate(shot["lastName"], (x + 1, y - 2), ax=ax, fontsize=12)
            elif shot.shot_outcome == 'Missed':
                pitch.scatter(x, y, alpha=1, s=500, color='red', ax=ax)
                pitch.annotate(shot["lastName"], (x + 1, y - 2), ax=ax, fontsize=12)
        goal_legend = plt.Line2D([0], [0], color="green", lw=4)
        accurate_legend = plt.Line2D([0], [0], color="blue", lw=4)
        missed_legend = plt.Line2D([0], [0], color="red", lw=4)
        ax.legend([goal_legend, accurate_legend, missed_legend], ['Goal', 'Accurate', 'Missed'], loc='upper right')
        match_name = plot_df['matchName'].iloc[0]
        fig.suptitle(match_name, fontsize=24)
        fig.set_size_inches(12, 8)
        file_name = '{}_plot.jpg'.format(match_name)
        plt.savefig("figures/" + file_name)
        plt.close()

    def plot_passes(self, match_id):
        df_polish_mathes_events = self.polish_matches_events[(self.polish_matches_events.eventName == 'Pass') & (self.polish_matches_events.teamId == 13869)].rename(columns={'label': 'matchName'})
        df_polish_mathes_events = pd.merge(df_polish_mathes_events, self.polish_players_names, on='playerId')
        mask_poland = (df_polish_mathes_events.subEventName != "Throw-in") & (df_polish_mathes_events.matchId == match_id)
        df_passes = df_polish_mathes_events.loc[mask_poland, ['x', 'y', 'end_x', 'end_y', 'lastName', 'accurate']]
        names = df_passes['lastName'].unique()
        pitchLengthX = 120
        pitchWidthY = 80
        pitch = Pitch(line_color='black', pad_top=20, pitch_length=pitchLengthX, pitch_width=pitchWidthY)
        fig, axs = pitch.grid(ncols=4, nrows=4, grid_height=0.85, title_height=0.06, axis=False,
                              endnote_height=0.04, title_space=0.04, endnote_space=0.01)
        df_passes['x'] = df_passes['x'] / 100 * pitchLengthX
        df_passes['y'] = df_passes['y'] / 100 * pitchWidthY
        df_passes['end_x'] = df_passes['end_x'] / 100 * pitchLengthX
        df_passes['end_y'] = df_passes['end_y'] / 100 * pitchWidthY
        for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
            player_df = df_passes.loc[df_passes["lastName"] == name]
            total_passes = len(player_df)
            accurate_passes = len(player_df[player_df['accurate'] == 1])
            accuracy_share = accurate_passes / total_passes * 100
            ax.text(60, -10, f"{name} ({accuracy_share:.1f}%)", ha='center', va='center', fontsize=14)
            for idx, row in player_df.iterrows():
                arrow_color = "green" if row['accurate'] == 1 else "red"
                pitch.arrows(row.x, row.y, row.end_x, row.end_y, color=arrow_color, ax=ax, width=1)
                pitch.scatter(row.x, row.y, alpha=0.2, s=50, color=arrow_color, ax=ax)
        for ax in axs['pitch'][-1, 16 - len(names):]:
            ax.remove()
        axs['title'].text(0.5, 0.5, f'{self.country_name1} passes against Senegal', ha='center', va='center', fontsize=30)
        plt.savefig("figures/" + f'{self.country_name1}_passes.jpg')
        plt.close()

    def plot_offensive_passes(self, match_id):
        df_polish_mathes_events = self.polish_matches_events[(self.polish_matches_events.eventName == 'Pass') & (self.polish_matches_events.teamId == 13869)].rename(columns={'label': 'matchName'})
        df_polish_mathes_events = pd.merge(df_polish_mathes_events, self.polish_players_names, on='playerId')
        mask_poland = (df_polish_mathes_events.subEventName != "Throw-in") & (df_polish_mathes_events.matchId == match_id)
        df_passes = df_polish_mathes_events.loc[mask_poland, ['x', 'y', 'end_x', 'end_y', 'lastName', 'accurate']]
        df_passes = df_passes[df_passes.end_x > df_passes.x]
        names = df_passes['lastName'].unique()
        pitchLengthX = 120
        pitchWidthY = 80
        pitch = Pitch(line_color='black', pad_top=20, pitch_length=pitchLengthX, pitch_width=pitchWidthY)
        fig, axs = pitch.grid(ncols=4, nrows=4, grid_height=0.85, title_height=0.06, axis=False,
                              endnote_height=0.04, title_space=0.04, endnote_space=0.01)
        df_passes['x'] = df_passes['x'] / 100 * pitchLengthX
        df_passes['y'] = df_passes['y'] / 100 * pitchWidthY
        df_passes['end_x'] = df_passes['end_x'] / 100 * pitchLengthX
        df_passes['end_y'] = df_passes['end_y'] / 100 * pitchWidthY
        for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
            player_df = df_passes.loc[df_passes["lastName"] == name]
            total_passes = len(player_df)
            accurate_passes = len(player_df[player_df['accurate'] == 1])
            accuracy_share = accurate_passes / total_passes * 100
            ax.text(60, -10, f"{name} ({accuracy_share:.1f}%)", ha='center', va='center', fontsize=14)
            for idx, row in player_df.iterrows():
                arrow_color = "green" if row['accurate'] == 1 else "red"
                pitch.arrows(row.x, row.y, row.end_x, row.end_y, color=arrow_color, ax=ax, width=1)
                pitch.scatter(row.x, row.y, alpha=0.2, s=50, color=arrow_color, ax=ax)
        for ax in axs['pitch'][-1, 16 - len(names):]:
            ax.remove()
        axs['title'].text(0.5, 0.5, f'{self.country_name1} offensive passes against Senegal', ha='center', va='center', fontsize=30)
        plt.savefig("figures/" + f'{self.country_name1}_offensive_passes.jpg')
        plt.close()

    def plot_passing_network(self, match_id):
        df = self.polish_matches_events[(self.polish_matches_events.matchId == match_id) & (self.polish_matches_events.teamId == 13869) & (self.polish_matches_events.eventName == 'Pass') & (self.polish_matches_events.matchPeriod == '1H')].rename(columns={'label': 'matchName'})
        df['next_player'] = df['playerId'].shift(-1)
        df.loc[df['accurate'] != 1, 'next_player'] = None
        df = df[df['next_player'].notnull()]
        df = df[['playerId', 'x', 'y', 'end_x', 'end_y', 'eventSec', 'next_player']]
        polish_sen = self.polish_players[(self.polish_players.matchId == match_id) & (self.polish_players.teamId == '13869')]
        polish_sen = polish_sen[['playerId', 'minute', 'lastName']]
        players = polish_sen[['playerId', 'lastName']].drop_duplicates()
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
        scatter_df = pd.DataFrame()
        for i, name in enumerate(df_pass["lastName"].unique()):
            passx = df_pass.loc[df_pass["lastName"] == name]["x"].to_numpy()
            recx = df_pass.loc[df_pass["lastName_next"] == name]["end_x"].to_numpy()
            passy = df_pass.loc[df_pass["lastName"] == name]["y"].to_numpy()
            recy = df_pass.loc[df_pass["lastName_next"] == name]["end_y"].to_numpy()
            scatter_df.at[i, "lastName"] = name
            scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
            scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
            scatter_df.at[i, "no"] = df_pass.loc[df_pass["lastName"] == name].count().iloc[0]
        scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)
        df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["lastName"], x["lastName_next"]])), axis=1)
        lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
        lines_df.rename({'x': 'pass_count'}, axis='columns', inplace=True)
        pitch = Pitch(line_color='grey', pitch_length=pitchLengthX, pitch_width=pitchWidthY)
        fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,endnote_height=0.02, title_space=0.02, endnote_space=0.01)
import re

import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup


class ElectionResultsFilepath:
    OAKLAND_COUNTY = 'G:/election_data/OaklandCountyElectionResults/'


class MajorParty:
    DEM: str = 'DEM'
    REP: str = 'REP'


class ProposalOption:
    YES: str = 'Yes'
    NO: str = 'No'


class _Office22Statewide:
    GOVERNOR = dict(key='124')
    SECRETARY_OF_STATE = dict(key='71')
    ATTORNEY_GENERAL = dict(key='28')
    STATE_BOARD_OF_EDUCATION = dict(key='161')
    SUPREME_COURT = dict(key='178')


class _Office24Statewide:
    POTUS = dict(key='124')
    STATE_BOARD_OF_EDUCATION = dict(key='161')
    SUPREME_COURT = dict(key='178')
    SUPREME_COURT_SPECIAL = dict(key='262')


class _Office24County:
    EXECUTIVE = dict(key='272')
    PROSECUTOR = dict(key='368')
    SHERIFF = dict(key='244')
    WATER_RESOURCES_COM = dict(key='216')
    WALLED_LAKE_SCHOOL_BOARD = dict(key='141')
    CLERK = dict(text='Clerk and Register of Deeds')


class _Office24Multi:
    SCHOOL_BOARD = dict(text=lambda x: 'Board Member' in str(x) and 'School' in str(x))
    LIBRARY_BOARD = dict(text=lambda x: 'Board Member' in str(x) and 'Library' in str(x))
    LEGISLATURE = dict(text=lambda x: str(x).startswith('Representative in State Legislature'))
    CONGRESS = dict(text=lambda x: str(x).startswith('Representative in Congress'))
    COUNTY_COM = dict(text=lambda x: str(x).startswith('County Commissioner'))


class Office22:
    Statewide = _Office22Statewide


class Office24:
    Statewide = _Office24Statewide
    County = _Office24County
    Multi = _Office24Multi


def _find_millage_with_text(topic: str) -> dict:
    return dict(text=lambda x: 'Millage' in str(x) and (topic in str(x) or re.search(topic, x, re.I)))


def _find_millage_without_text(topic: str) -> dict:
    return dict(text=lambda x: 'Millage' in str(x) and topic not in str(x))


def read_file(year: int | str) -> BeautifulSoup:
    page = BeautifulSoup(open(f'{ElectionResultsFilepath.OAKLAND_COUNTY}/detail{year}.xml').read(), 'xml')
    results_elem = page.find('ElectionResult')
    return results_elem


def _aggregate_2024(results: pd.DataFrame) -> pd.DataFrame:
    parties = results.groupby(['precinct', 'party'], as_index=False).votes.sum()
    totals = results.groupby('precinct', as_index=False).votes.sum()
    results = parties.merge(totals, on='precinct', suffixes=('', 'Share')).rename(columns=dict(votesShare='voteShare'))
    results.voteShare = results.votes / results.voteShare
    results = results.drop(columns='votes')
    return results


def filter_office(results_elem: BeautifulSoup, office: dict, party: str = None) -> pd.DataFrame:
    data = []
    for contest_elem in results_elem.find_all('Contest', office):
        for candidate_elem in contest_elem.find_all('Choice'):
            for precinct_elem in candidate_elem.find_all('Precinct'):
                data.append(dict(
                    precinct=precinct_elem['name'].strip(),
                    votes=precinct_elem['votes'].strip(),
                    candidate=candidate_elem.get('text'),
                    party=candidate_elem.get('party'),
                ))

    results = pd.DataFrame(data)
    results.votes = results.votes.map(int)
    if party:
        results = results[results.party.isin((MajorParty.DEM, MajorParty.REP))].drop(columns='candidate')
        results = _aggregate_2024(results)
        results = results[results.party == party].copy()
    else:
        results.candidate = results.candidate.str.title()
        results = results.drop(columns='party')
    return results


def filter_millage(results_elem: BeautifulSoup, millage: dict) -> pd.DataFrame:
    data = []
    for millage_elem in results_elem.find_all('Contest', millage):
        for option_elem in millage_elem.find_all('Choice'):
            for precinct_elem in option_elem.find_all('Precinct'):
                data.append(dict(
                    precinct=precinct_elem['name'].strip(),
                    votes=precinct_elem['votes'],
                    party=option_elem['text'],  # option: YES/NO
                ))

    results = pd.DataFrame(data)
    if not len(results):
        return results
    results.votes = results.votes.map(int)

    return _aggregate_2024(results)


def _assign_parties_to_nonpartisan(results: pd.DataFrame, progressives: tuple, conservatives: tuple) -> pd.DataFrame:
    results.loc[results.candidate.isin(progressives), 'party'] = MajorParty.DEM
    results.loc[results.candidate.isin(conservatives), 'party'] = MajorParty.REP
    return results.dropna(subset=['party'])


def filter_walled_lake_school_board(results_elem: BeautifulSoup) -> pd.DataFrame:
    results = filter_office(results_elem, Office24.County.WALLED_LAKE_SCHOOL_BOARD)
    progressives = ('Susie Crafton', 'Marc A. Siegler', 'Ron Lippitt', 'Michael Smith')
    conservatives = ('Tricia Auten', 'Rebecca Behrends', 'Lisa West', 'Steve Rix')
    return _aggregate_2024(_assign_parties_to_nonpartisan(results, progressives, conservatives))


def filter_supreme_court(results_elem: BeautifulSoup) -> pd.DataFrame:
    results = pd.concat((
        filter_office(results_elem, office)
        for office in (Office24.Statewide.SUPREME_COURT, Office24.Statewide.SUPREME_COURT_SPECIAL)
    ))
    progressives = ('Kyra Harris Bolden', 'Kimberly Ann Thomas')
    conservatives = ('Andrew Fink', 'Patrick William O\'Grady')
    return _aggregate_2024(_assign_parties_to_nonpartisan(results, progressives, conservatives))


def analyze_walled_lake_school_board(results_elem: BeautifulSoup) -> pd.DataFrame:
    party = MajorParty.REP
    results = filter_walled_lake_school_board(results_elem)
    results = results[results.party == party].copy()
    results_boe = filter_office(results_elem, Office24.Statewide.STATE_BOARD_OF_EDUCATION, party)
    results_potus = filter_office(results_elem, Office24.Statewide.POTUS, party)

    suffixes = ('Parental Rights Candidates', 'Trump')

    results = results.merge(results_potus, on='precinct', suffixes=suffixes).melt(['precinct'], [
        f'voteShare{i}' for i in suffixes], '', 'voteShare')
    results[''] = results[''].str.slice(9)

    results = results.merge(results_boe.rename(columns=dict(voteShare='voteShareBoe')), on='precinct')

    ax = sns.lmplot(results, x='voteShareBoe', y='voteShare', hue='', markers='.', hue_order=suffixes, palette=(
        'orange', 'purple'))
    ax.fig.suptitle('WALLED LAKE SCHOOL BOARD\nPerformance of Parental Rights Candidates')
    ax.set_xlabels('STATE BOARD OF ED. Vote Share')
    ax.set_ylabels('PARENTAL RIGHTS CANDIDATES Combined Vote Share')
    ax.figure.set_size_inches(10, 6)
    ax.figure.savefig('img/2024 - Walled Lake School Board - Parental Rights Performance.png')
    ax.figure.clf()

    return results


def analyze_president_vs_congress(results_elem: BeautifulSoup) -> pd.DataFrame:
    party = MajorParty.REP

    results = filter_office(results_elem, Office24.Multi.LEGISLATURE, party)
    results_congress = filter_office(results_elem, Office24.Multi.CONGRESS, party)
    results_potus = filter_office(results_elem, Office24.Statewide.POTUS, party)

    results = results.merge(results_congress, on='precinct', suffixes=('Leg', 'Cong'))
    results = results.melt(['precinct'], ['voteShareLeg', 'voteShareCong'], 'Office', 'voteShare')
    results.Office = results.Office.str.slice(9)

    results = results.merge(results_potus.rename(columns=dict(voteShare='voteSharePotus')), on='precinct')

    results_potus = results_potus[results_potus.precinct.isin(results.precinct)].copy()
    results_potus['Office'] = 'POTUS'
    results_potus['voteSharePotus'] = results_potus.voteShare.copy()
    results = pd.concat((results, results_potus))

    ax = sns.lmplot(results, x='voteSharePotus', y='voteShare', hue='Office', markers='.', hue_order=(
        'POTUS', 'Cong', 'Leg'), palette=('grey', 'purple', 'orange'))
    ax.fig.suptitle('Performance of Congressional/Legislative Candidates')
    ax.set_xlabels('TRUMP Vote Share')
    ax.set_ylabels('CONGRESS/LEGISLATURE Aggregate Vote Share')
    ax.figure.set_size_inches(10, 6)
    ax.figure.savefig('img/2024 - President vs. Congress.png')
    ax.figure.clf()

    return results


def analyze_countywide_offices(results_elem: BeautifulSoup) -> pd.DataFrame:
    party = MajorParty.REP

    results = filter_office(results_elem, Office24.County.EXECUTIVE, party)
    results_prosecutor = filter_office(results_elem, Office24.County.PROSECUTOR, party)
    results_sheriff = filter_office(results_elem, Office24.County.SHERIFF, party)
    results_clerk = filter_office(results_elem, Office24.County.CLERK, party)
    results_water = filter_office(results_elem, Office24.County.WATER_RESOURCES_COM, party)
    results_potus = filter_office(results_elem, Office24.Statewide.POTUS, party)

    results = (
        results
        .merge(results_prosecutor, on='precinct', suffixes=('', 'Prosecutor'))
        .merge(results_sheriff, on='precinct', suffixes=('', 'Sheriff'))
        .merge(results_clerk, on='precinct', suffixes=('', 'Clerk'))
        .merge(results_water, on='precinct', suffixes=('Executive', 'WaterCom'))
    )
    results = results.melt(
        ['precinct'],
        ['voteShareExecutive', 'voteShareProsecutor', 'voteShareSheriff', 'voteShareClerk', 'voteShareWaterCom'],
        'Office', 'voteShare'
    )
    results.Office = results.Office.str.slice(9)
    results = results.merge(results_potus.rename(columns=dict(voteShare='voteSharePotus')), on='precinct')

    results_potus = results_potus[results_potus.precinct.isin(results.precinct)].copy()
    results_potus['Office'] = 'POTUS'
    results_potus['voteSharePotus'] = results_potus.voteShare.copy()
    results = pd.concat((results, results_potus))

    ax = sns.lmplot(results, x='voteSharePotus', y='voteShare', hue='Office', markers='.')
    ax.fig.suptitle('OAKLAND COUNTY REPUBLICANS\nPerformance of Countywide Republicans vs. Trump')
    ax.set_xlabels('TRUMP Vote Share')
    ax.set_ylabels('COUNTYWIDE REPUBLICAN CANDIDATE Vote Share')
    ax.figure.set_size_inches(10, 6)
    ax.figure.savefig('img/2024 - Oakland County Republicans.png')
    ax.figure.clf()

    return results


def analyze_millage_vs_dem(results_elem: BeautifulSoup, party: str) -> pd.DataFrame:
    millage_topics = (
        'Police',
        'Senior Services',
        'Fire',
        'School|Educ',
        'Library',
        'Parks|Recreation|Playground|Path',
        'Road|Street',
        'Public Transportation',
    )

    dfs = []
    millage_topics_with_millages = []
    for topic in millage_topics:
        df = filter_millage(results_elem, _find_millage_with_text(topic))
        if not len(df):
            continue
        df = df[df.party == ProposalOption.YES].copy()
        df['Millage Topic'] = topic
        dfs.append(df)
        millage_topics_with_millages.append(topic)

    results = pd.concat(dfs)

    results_state = filter_office(results_elem, Office24.Statewide.STATE_BOARD_OF_EDUCATION, party)
    results = results.merge(results_state, on='precinct', suffixes=('Millage', 'State'))

    ax = sns.lmplot(
        results, x='voteShareState', y='voteShareMillage', hue='Millage Topic', hue_order=millage_topics_with_millages,
        markers='.', palette='hls',
    )
    ax.fig.suptitle(f'2024 OAKLAND COUNTY MILLAGES\nMillage Performance vs. {party} Vote Share')
    ax.set_xlabels(f'{party} Vote Share')
    ax.set_ylabels('Millage YES Vote Share')
    ax.figure.set_size_inches(12, 8)
    ax.figure.savefig('img/2024 - Oakland County Millages.png')
    ax.figure.clf()

    return results


def main() -> None:
    results_elem = read_file(2024)
    analyze_walled_lake_school_board(results_elem)
    analyze_countywide_offices(results_elem)
    analyze_millage_vs_dem(results_elem, MajorParty.DEM)
    return


if __name__ == '__main__':
    main()

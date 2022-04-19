import numpy
import pandas as pd
import math
import time
import os
from bs4 import BeautifulSoup
import pickle
import statsmodels
from statsmodels import api as sm
import scipy

start = ''

def check_for_folders():
    folders = ['html_files', 'df_files', 'model_files']
    for folder in folders:
        if not os.path.exists(folder): os.makedirs(folder)

def paste_college_html():
    check_for_folders()
    temp_path = os.path.join(start, 'html_files', 'temp.html')
    os.system(f'xsel -b >> {temp_path}')
    with open(temp_path) as f:
        soup = BeautifulSoup(f, features='lxml')
    timestr = time.strftime("%Y%m%d")
    def has_class(tag):
        return tag['class'] == 'school-name-text'
    #college = root.find(".//[@class='school-name-text']").text
    college = soup.find('h1', class_='school-name-text').string
    moniker = college.replace(' ', '_') + '_' + timestr
    html_filename = moniker + '.html'
    html_filepath = os.path.join(start, 'html_files', html_filename)
    os.replace(temp_path, html_filepath)
    return moniker

def html_to_pickles(moniker):
    html_filename = moniker + '.html'
    html_filepath = os.path.join(start, 'html_files', html_filename)
    with open(html_filepath) as f:
        soup = BeautifulSoup(f, features='lxml')
    df = make_df(soup)
    pickle_filename = moniker + '.pickle'
    pickle_filepath = os.path.join(start, 'df_files', pickle_filename)
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(df, f)
    rate = (0.0 + len(df[df['accepted']==0]))/len(df)
    model_filepath = os.path.join(start, 'model_files', pickle_filename)
    college_info = {'coefs': make_model(df), 'rate_change_p': rate_change_p(soup, rate)} 
    with open(model_filepath, 'wb') as f:
        pickle.dump(college_info, f)

def paste_college():
    html_to_pickles(paste_college_html())

def make_df(soup):
    accepted_groups = [str(x) for x in range(0,6)]
    denied_groups = [str(x) for x in range(9,15)]
    df = pd.DataFrame(columns=['group_num','point_num','x','y','SAT','GPA'
        ,'accepted'])
    df = df.astype({'group_num': str, 'point_num': str, 'x': float, 'y': float,
        'SAT':int, 'GPA':float, 'accepted':int})
    #accepted is an int for ease of graphing, it's just 0 or 1
    def get_coordinates(transform):
        #Looks like "translate(562.66667,210.5690)"
        transform = transform['transform']
        x_pos = 10
        y_pos = transform.find(',') + 1
        x = float(transform[x_pos:y_pos - 1])
        y = float(transform[y_pos:-1])
        return {'x':x, 'y':y}
    groups = soup.find('g', class_='nv-groups')
    for group in groups:
        #class="nv-group nv-series-#"
        group_num = group['class'][1][10:]
        if group_num in accepted_groups: accepted = 1
        elif group_num in denied_groups: accepted = 0
        else: continue
        for point in group:
            #class="nv-point nv-point-#"
            point_num = point['class'][1][9:]
            transform = get_coordinates(point)
            df = df.append({'group_num':group_num,'point_num':point_num,
                'x': transform['x'], 'y': transform['y'], 'accepted':accepted},
                ignore_index=True)
    df = remove_duplicates(df)
    def get_line(extremum, axis):
        return soup.find('g', class_=f'nv-axis{extremum}-{axis}')
    def get_extremum(extremum, axis):
        return float(get_line(extremum, axis).find('text').text)
    def convert(coordinate, axis):
        extremum = None
        if axis == 'x': extremum = 'Max'
        elif axis == 'y': extremum = 'Min'
        distance = get_coordinates(get_line(extremum,axis))[axis]
        if axis == 'y': coordinate = distance - coordinate
        return coordinate/distance*(get_extremum('Max',axis)-get_extremum('Min',
            axis))+get_extremum('Min',axis)
    def SAT(x):
        return round(convert(x,'x'),0)
    def GPA(y):
        return convert(y,'y') #(1.0-y/height)*5.0
    for index in df.index:
        df.loc[index,'SAT'] = SAT(df.loc[index,'x'])
        df.loc[index,'GPA'] = GPA(df.loc[index,'y'])
    return df

def remove_duplicates(df):
    df = df.copy()
    def first_repeat(index):
        x = group.iloc[index]['x']
        y = group.iloc[index]['y']
        identicals_of_first = group[(group['x'] == x) & (group['y'] == y)]
        if len(identicals_of_first) > 1:
            #[0] would be the first point itself
            return identicals_of_first.index.to_list()[1]
        else: return None
    for group in df.groupby('group_num'):
        group_num = group[0]
        group = group[1]
        cutoff = first_repeat(0)
        if not cutoff:
            continue
        for index in range(1,3):
            if not cutoff + index == first_repeat(index):
                raise Exception()
            #print(group_num, "group:", len(identicals_of_first),
            #        "repeats found. First repeat at", first_repeat_index)
            #DataFrame.index makes it some kind of array of the indices of the df
        all_indices = group.index.to_list()
        indices_to_delete = range(cutoff, all_indices[-1] + 1)
        df = df.drop(indices_to_delete)
        #else: print(group_num, " group: No repeats found.")
    df = df.reset_index(drop=True)
    return df

def make_model(df):
    logit_res = None
    def make_coefs(coefs):
        logit_mod = sm.Logit(df['accepted'],
                statsmodels.tools.add_constant(df[coefs]))
        nonlocal logit_res
        logit_res = logit_mod.fit()
        return logit_res.params
    coefs = make_coefs(['SAT', 'GPA'])
    new_coefs = []
    for key, coef in coefs.items():
        if not key == 'const':
            margin = 1.96 * logit_res.bse[key]
            if coef - margin > 0.0 and coef + margin > 0.0:
                new_coefs.append(key)
    if len(new_coefs) < 2: coefs = make_coefs(new_coefs)
    return coefs

def remove_latest_year(soup):
    #The first of this class to show up is the leftmost bar, the latest year
    latest_year = soup.find('div', class_='multibar-block-container')
    latest_year.decompose()

def rate_change_p(soup, rate):
    '''
    Returns p-value for whether the rate has changed
    '''
    remove_latest_year(soup)
    acceptances = 0
    applicants = 0
    stats = soup.find_all('div', class_='multibar-bar-block-label')
    def num(string):
        return int(string[:string.find(' ')])
    for stat in stats:
        stat = stat.text
        if stat.endswith(' Accepted'): acceptances += num(stat)
        elif stat.endswith(' Applied'): applicants += num(stat)
    cdf = scipy.stats.binom.cdf(acceptances, applicants, rate)
    if cdf > 0.5: cdf = 1 - cdf
    return cdf

def make_model_complex_ver(df):
    logit_mod = sm.Logit(df['accepted'],
            statsmodels.tools.add_constant(df[['SAT','GPA']]))
    logit_res = logit_mod.fit()
    coefs = {'const':None, 'SAT':None, 'GPA':None}
    for key in coefs:
        coef = logit_res.params[key]
        margin = 1.96 * logit_res.bse[key]
        coefs[key] = [coef - margin, coef, coef + margin]
        if not key == 'const':
            for i, num in enumerate(coefs[key]):
                if num < 0:
                    coefs[key][i] = 0
    return coefs

def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def file_stuff(folder):
    loc = os.path.join(start, folder)
    files = os.listdir(loc)
    return loc, files

def get_college(college_name, folder):
    loc, files = file_stuff(folder)
    def starts_with(item):
        return item.startswith(college_name)
    college_files = filter(starts_with, files)
    #Get the latest file by date
    filepath = os.path.join(loc, max(college_files))
    if filepath.endswith('.html'):
        with open(filepath) as f:
            return BeautifulSoup(f, features='lxml')
    return read_pickle(filepath)

def get_colleges(folder):
    colleges = {}
    for college in get_college_names():
        colleges[college] = get_college(college, folder)
    return colleges

def get_college_names():
    colleges = []
    loc, files = file_stuff('model_files')
    for file in files:
        #Get just the college name
        colleges.append(file[:-16])
    return colleges

def update_pickles():
    for filename in os.listdir(os.path.join(start, 'html_files')):
        html_to_pickles(filename[:-5])
        print(filename, 'done') 

def update_to_rate_change_p(moniker):
    with open(os.path.join(start, 'df_files', moniker+'.pickle'), 'rb') as f:
        df = pickle.load(f)
    rate = (0.0 + len(df[df['accepted']==1]))/len(df)
    with open(os.path.join(start, 'html_files', moniker + '.html')) as f:
        soup = BeautifulSoup(f, features='lxml')
    p = rate_change_p(soup, rate)
    with open(os.path.join(start, 'model_files', moniker+'.pickle'), 'rb') as f:
        old = pickle.load(f)
    if old.get('recently_accepted'):
        old.pop('recently_accepted')
    old['rate_change_p'] = p
    with open(os.path.join(start, 'model_files', moniker+'.pickle', 'wb') as f:
        pickle.dump(old, f)

def mass_update_to_rate_change_p():
    for filename in os.listdir(os.path.join(start, 'model_files')):
        print(filename[:-16])
        moniker = filename[:-7]
        update_to_rate_change_p(moniker)

def predict(college, stats):
    college = college.replace(' ','_')
    params = get_college(college, 'model_files')['coefs']
    log_odds = 0.0
    missing = []
    for param, value in params.iteritems():
        if param == 'const':
            log_odds += value
            continue
        stat = stats.get(param)
        if stat:
            log_odds += value*stat
        else:
            missing.append(param)
    chance = math.nan
    if len(missing) == 0:
        sigmoid = lambda x: 1 / (1 + numpy.exp(-x))
        chance = sigmoid(log_odds)
    return (chance, missing)

def all_college_predictions(sat, gpa):
    colleges = get_college_names()
    ddf = {'name':colleges, 'chance':[], 'missing':[], 'info':[],
            'recently_accepted':[]}
    for college in colleges:
        chance, missing = predict(college, {'SAT':sat,'GPA':gpa})
        ddf['chance'].append(chance)
        ddf['missing'].append(missing)
        college_info = get_college(college, 'model_files')
        if len(college_info['coefs']) == 1:
            ddf['info'].append('const_only')
        else: ddf['info'].append(math.nan)
        ddf['recently_accepted'].append(college_info['recently_accepted'])
    return pd.DataFrame(ddf).set_index('name')

def accepted_df():
    df = make_df()
    accepted_groups = [str(x) for x in range(0,6)]
    accepted_df = df[df['group_num'].isin(accepted_groups)]
    return accepted_df

def block_maker(df, col, increment):
    return numpy.arange(df[col].min(),df[col].max(),increment)

def blocked_chances(df, col, increment):
    #bunches = [x*bunch_num for x in range(int(math.ceil(len(df)/int(bunch_num))))]
    blocks = block_maker(df, col, increment)
    chances = {col:[], 'accepted':[]}
    for bunch_start in blocks:
        #bunch = df.loc[bunch_start:bunch_start + bunch_num]
        bunch = df[(df[col]>=bunch_start)&(df[col]<bunch_start+increment)]
        if len(bunch) > 0:
            chances[col].append(bunch[col].mean())
            chances['accepted'].append(bunch['accepted'].mean())
    return chances

def avgs(df, col, increment):
    blocks = block_maker(df, col, increment)
    averages = []
    for bunch_start in blocks:
        bunch = df[(df[col]>=bunch_start)&(df[col]<bunch_start+increment)]
        if len(bunch) > 0:
            averages.append(bunch[col].mean())
        else:
            averages.append(numpy.NaN)
    return averages

def blocked_2d_chances(df, sat_increment, gpa_increment):
    sat_blocks = block_maker(df, 'SAT', sat_increment)
    gpa_blocks = block_maker(df, 'GPA', gpa_increment)
    #gpa_avgs = numpy.empty_like(gpa_blocks)
    #for blocks, avgs in (sat_blocks, sat_avgs), (gpa_blocks, gpa_avgs):
    #    for index, start in enumerate(blocks):
    #        bunch = df[
    chances = numpy.empty([len(sat_blocks),len(gpa_blocks)])
    chances[:] = numpy.NaN
    for sat_index, sat_bunch_start in enumerate(sat_blocks):
        for gpa_index, gpa_bunch_start in enumerate(gpa_blocks):
            bunch = df[(df['SAT']>=sat_bunch_start)
                    &(df['SAT']<sat_bunch_start+sat_increment)
                    &(df['GPA']>=gpa_bunch_start)
                    &(df['GPA']<gpa_bunch_start+gpa_increment)]
            if len(bunch) > 0:
                chances[sat_index,gpa_index] = bunch['accepted'].mean()
    return chances

def blocked_2d_totals(df, sat_increment, gpa_increment):
    sat_blocks = block_maker(df, 'SAT', sat_increment)
    gpa_blocks = block_maker(df, 'GPA', gpa_increment)
    #gpa_avgs = numpy.empty_like(gpa_blocks)
    #for blocks, avgs in (sat_blocks, sat_avgs), (gpa_blocks, gpa_avgs):
    #    for index, start in enumerate(blocks):
    #        bunch = df[
    chances = numpy.empty([len(sat_blocks),len(gpa_blocks)])
    chances[:] = numpy.NaN
    for sat_index, sat_bunch_start in enumerate(sat_blocks):
        for gpa_index, gpa_bunch_start in enumerate(gpa_blocks):
            bunch = df[(df['SAT']>=sat_bunch_start)
                    &(df['SAT']<sat_bunch_start+sat_increment)
                    &(df['GPA']>=gpa_bunch_start)
                    &(df['GPA']<gpa_bunch_start+gpa_increment)]
            chances[sat_index,gpa_index] = len(bunch)
    return chances

def predicted_blocks(df, sat_increment, gpa_increment, logit_res_func):
    sat_avgs = avgs(df, 'SAT', sat_increment)
    gpa_avgs = avgs(df, 'GPA', gpa_increment)
    chances = numpy.empty([len(sat_avgs),len(gpa_avgs)])
    chances[:] = numpy.NaN
    for s_i, sat in enumerate(sat_avgs):
        for g_i, gpa in enumerate(gpa_avgs):
            pred = logit_res_func(sat, gpa)
            #pred = logit_res.predict(pd.DataFrame({'const': 1.0, 'SAT': [sat],
            #    'GPA': [gpa]}))
            chances[s_i, g_i] = pred
    return chances

def check_avg():
    df = accepted_df()
    print('SAT', df['SAT'].mean())
    print('GPA', df['GPA'].mean())

def main():
    pass

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import os
import pickle
import vitaldb
import random
from pyvital import arr
import time
import multiprocessing

def data_loader(caseid):
        
    opstart = df_cases[df_cases['caseid']==caseid]['opstart'].values[0]
    opend = df_cases[df_cases['caseid']==caseid]['opend'].values[0]
    
    demo = df_cases[df_cases['caseid']==caseid][['age', 'sex', 'weight', 'height', 'asa']].values
    
    MINUTES_AHEAD = 5*60
    SRATE = 100
    INSEC = 30  # input 길이 (논문과 같음)
    SW_SIZE = 5  # sliding window size (논문에 따로 정의되어 있지 않음)

    vals = vitaldb.load_case(caseid, [ECG, PPG, ART, CO2, MBP], 1/SRATE)
    vals[:,2] = arr.replace_undefined(vals[:,2])
    vals = vals[opstart * SRATE:opend * SRATE]
    
    # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
    x = []
    y = []
    c = []
    a = []
    for i in range(0, len(vals) - (SRATE * (INSEC + MINUTES_AHEAD) + 1), SRATE * SW_SIZE):
        segx = vals[i:i + SRATE * INSEC, :4]  
        segy = vals[i + SRATE * (INSEC + MINUTES_AHEAD) + 1, 4]
        
        if segy < 20 or segy > 200:
            continue

        '''
        # maic 대회 참고하여 전처리 
        if np.mean(np.isnan(segx)) > 0 or \
            np.mean(np.isnan(segy)) > 0 or \
            np.max(segy) > 200 or np.min(segy) < 20 or \
            np.max(segy) - np.min(segy) < 30 or \
            (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
            i += SRATE  # 1 sec 씩 전진
            continue
        '''    
        x.append(segx)
        y.append(segy)
        c.append(caseid)
        a.append(demo)
                
    if len(x) > 0:
        print(caseid)
        ret = (np.array(x), np.array(y), np.array(c), np.array(a)) 
        pickle.dump((ret), open(f"/home/seonga/md_hypo/minutes5_reg/{caseid}_vf.pkl", "wb"))


if __name__ == '__main__':
    
    df_trks = pd.read_csv('https://api.vitaldb.net/trks')  # 트랙 목록
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # 임상 정보

    ECG = 'SNUADC/ECG_II'
    PPG = 'SNUADC/PLETH'
    ART = 'SNUADC/ART'
    MBP = 'Solar8000/ART_MBP'
    CO2 = 'Primus/CO2'  # 62.5hz 로 다른 waveform 과 주파수가 다름 

    caseids = list(
    set(df_trks[df_trks['tname'] == MBP]['caseid']) &
    set(df_trks[df_trks['tname'] == CO2]['caseid']) &
    set(df_trks[df_trks['tname'] == ART]['caseid']) &
    set(df_trks[df_trks['tname'] == ECG]['caseid']) &
    set(df_trks[df_trks['tname'] == PPG]['caseid']))
    
    df_cases = df_cases[(df_cases['caseid'].isin(caseids))&(df_cases['age']>=18)&(df_cases['death_inhosp']!=1)]
    caseids = list(df_cases['caseid'].values)

    caseids = list(set(caseids) - set(already_caseids))
    print(len(caseids))
    
    start_time = time.time()
    n_process = 20

    manager = multiprocessing.Manager() 
    d = manager.dict() # shared dictionary

    pool = multiprocessing.Pool(processes=n_process)
    pool.map(data_loader, caseids)

    pool.close()
    pool.join()
    
    print("=== %s seconds ===" % (time.time() - start_time))   # 5개 266초  

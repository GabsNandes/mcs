import numpy as np
import os
import pandas as pd
from tqdm import tqdm

inicio = '202301'
fim = '202312'

dir_path = 'data/processed/lst'

meses = pd.period_range(start=inicio, end=fim, freq='M').strftime('%Y%m')

for mes in meses:
    arquivos = [f for f in os.listdir(dir_path) if f.startswith(mes) and f.endswith('.npz') and len(f) == 12]
    
    if not arquivos:
        continue

    avgs, mins, maxs = [], [], []

    for arquivo in tqdm(arquivos, desc=f"Processando {mes}"):
        data = np.load(os.path.join(dir_path, arquivo))
        avgs.append(data['avg'])
        mins.append(data['min'])
        maxs.append(data['max'])

    avgs = np.array(avgs)
    mins = np.array(mins)
    maxs = np.array(maxs)

    media_mensal = np.nanmean(avgs, axis=0)
    maxima_mensal = np.nanmax(maxs, axis=0)
    minima_mensal = np.nanmin(mins, axis=0)
    media_maximas = np.nanmean(maxs, axis=0)
    media_minimas = np.nanmean(mins, axis=0)

    np.savez_compressed(os.path.join(dir_path, f'{mes}.npz'), avg=media_mensal, max=maxima_mensal, min=minima_mensal, max_max=media_maximas, min_avg=media_minimas)

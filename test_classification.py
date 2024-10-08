from src.downloads import download_TCGA_ctranspath, download_TCGA_moco, download_TCGA_phikon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd 
import numpy as np
from pathlib import Path

data_path_ctranspath = download_TCGA_ctranspath()
data_path_moco = download_TCGA_moco()
data_path_phikon = download_TCGA_phikon()

csv_paths = Path('csv').glob('*.csv')

R = []
res_moco = {'mean': [], 'std': []}
res_ctranspath = {'mean': [], 'std': []}
res_phikon = {'mean': [], 'std': []}
tasks = []
np.random.seed(37)
for csv_path in csv_paths:
    tasks.append(csv_path.stem)
    df = pd.read_csv(csv_path)
    if len(df['label'].unique())==2:
        scoring = 'roc_auc'
    else:
        scoring = 'roc_auc_ovr'

    data_ctranspath = np.load(data_path_ctranspath, allow_pickle=True).item()
    data_moco = np.load(data_path_moco, allow_pickle=True).item()
    data_phikon = np.load(data_path_phikon, allow_pickle=True).item()
    
    label_dict = df.set_index('ID').to_dict()['label']
    wsis_ctranspath = set(data_ctranspath.keys()).intersection(set(label_dict.keys()))
    wsis_moco = set(data_moco.keys()).intersection(set(label_dict.keys()))
    wsis_phikon = set([k for k,v in data_phikon.items() if v is not None]).intersection(set(label_dict.keys()))

    y_ctranspath = np.array([label_dict[wsi] for wsi in wsis_ctranspath])
    X_ctranspath = np.array([data_ctranspath[wsi] for wsi in wsis_ctranspath])


    y_moco = np.array([label_dict[wsi] for wsi in wsis_moco])
    X_moco = np.array([data_moco[wsi] for wsi in wsis_moco])

    y_phikon = np.array([label_dict[wsi] for wsi in wsis_phikon])
    X_phikon = np.array([data_phikon[wsi] for wsi in wsis_phikon])

    skf = StratifiedKFold(n_splits=10)
    
    logreg = LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=37)
    scores_ctranspath = cross_val_score(logreg, X_ctranspath, y_ctranspath, cv=skf, scoring=scoring)
    scores_moco = cross_val_score(logreg, X_moco, y_moco, cv=skf, scoring=scoring)
    scores_phikon = cross_val_score(logreg, X_phikon, y_phikon, cv=skf, scoring=scoring)

    res_moco['mean'].append(scores_moco.mean())
    res_moco['std'].append(scores_moco.std())

    res_ctranspath['mean'].append(scores_ctranspath.mean())
    res_ctranspath['std'].append(scores_ctranspath.std())

    res_phikon['mean'].append(scores_phikon.mean())
    res_phikon['std'].append(scores_phikon.std())


columns = pd.MultiIndex.from_tuples([('GigaSSL + Moco', 'Mean'), ('GigaSSL + Moco', 'Std'), 
                                     ('GigaSSL + CTransPath', 'Mean'), ('GigaSSL + CTransPath', 'Std'), 
                                     ('GigaSSL + PhiKon', 'Mean'), ('GigaSSL + PhiKon', 'Std')])
# Create DataFrame
df = pd.DataFrame(list(zip(res_moco['mean'], res_moco['std'], 
                           res_ctranspath['mean'], res_ctranspath['std'], 
                           res_phikon['mean'], res_phikon['std'])),
                  index=tasks, columns=columns).round(3)
df.to_csv('results.csv', index=False)

print(df)


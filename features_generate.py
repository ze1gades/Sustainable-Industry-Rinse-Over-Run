import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")#отключение варнингов
pd.set_option('display.max_columns', None)
recipe = pd.read_csv('recipe_metadata.csv',
                     index_col=0)
recipe['num_phases'] = recipe.sum(axis=1)
train_values = pd.read_csv('train_values.csv', index_col=0, parse_dates=['timestamp'])
train_labels = pd.read_csv('train_labels.csv', index_col=0)
test_values = pd.read_csv('test_values.csv', index_col=0, parse_dates=['timestamp'])


def prep_metadata(df, recipe):  # Подготовка метаданных
    meta = df[['process_id', 'pipeline']].drop_duplicates().set_index('process_id')
    meta = pd.get_dummies(meta)
    if 'L12' not in meta.columns:
        meta['pipeline_L12'] = 0
    meta = pd.concat([meta, recipe], axis=1)
    return meta


def prep_phase_date(phase_name, df):  # Подготовка данных о фазе phase_name
    group = df.groupby('phase')
    target_phase = group.indices[phase_name]
    target_phase = df.loc[target_phase]
    target_phase.pop('phase')
    ts_cols = [
        'process_id',
        'supply_flow',
        'supply_pressure',
        'return_temperature',
        'return_conductivity',
        'return_turbidity',
        'return_flow',
        'tank_level_pre_rinse',
        'tank_level_caustic',
        'tank_level_acid',
        'tank_level_clean_water',
        'tank_temperature_pre_rinse',
        'tank_temperature_caustic',
        'tank_temperature_acid',
        'tank_concentration_caustic',
        'tank_concentration_acid',
    ]
    ts_df = target_phase[ts_cols].set_index('process_id')
    phase_time = ts_df.groupby('process_id').count().values[:, 0]
    for name in ts_cols:
        ts_df.rename(columns={name: phase_name + '_' + name}, inplace=True)
    ret_features = ts_df.groupby('process_id').agg(['min', 'max', 'mean', 'std', lambda x: x.tail(5).mean()])
    ret_features[phase_name + '_phase_time'] = phase_time
    return ret_features


def get_process_id_with_phase(phase_name, df):  # Получение 'process_id' процессов имеющих фазу phase_name
    group = df.groupby('phase')
    target_phase = group.indices[phase_name]
    target_phase = df.loc[target_phase]
    target_phase = np.array(list(target_phase.groupby('process_id').groups.keys()))
    return target_phase


phase_list = np.array(['acid', 'intermediate_rinse', 'caustic', 'pre_rinse'])
mask = np.array([8, 4, 2, 1])
phases_features = {a: prep_phase_date(a, train_values) for a in phase_list}

print('Train:')
for i in range(1, 16):  # Подготовка тренеровочных датафреймов
    phase_mask = (i & mask) != 0  # Получаем буллевскую маску: True - фаза должна присутствовать, False - фаза не должна
                                  # присутствовать
    inclusion_phase = phase_list[phase_mask]  # Получаем названия фаз, которые должны присутствовать
    ts_features = prep_metadata(train_values, recipe)
    for phase in inclusion_phase:
        ts_features = pd.concat([ts_features, phases_features[phase]], axis=1)
    ts_features.dropna(inplace=True)
    print(inclusion_phase, ts_features.shape)
    ts_features.to_csv('type_' + str(i) + '_train.csv', index='process_id')

phases_features = {a: prep_phase_date(a, test_values) for a in phase_list}
phases_pid = {a: get_process_id_with_phase(a, test_values) for a in phase_list}

print('Test:')
for i in range(1, 16):  # Подготовка тестовых датафреймов
    phase_mask = (i & mask) != 0  # Получаем буллевскую маску: True - фаза должна присутствовать, False - фаза не должна
                                  # присутствовать
    inclusion_phase = phase_list[phase_mask]  # Получаем названия фаз, которые должны присутствовать
    exclusion_phase = phase_list[np.logical_not(phase_mask)]  # Получаем названия фаз, которые должны отсутствовать
    exclusion_pid = np.array([])
    for j in range(len(exclusion_phase)):  # Создаем список 'process_id' на удаление
        exclusion_pid = np.unique(np.concatenate((exclusion_pid, phases_pid[exclusion_phase[j]]), axis=0))
    ts_features = prep_metadata(test_values, recipe)
    for phase in inclusion_phase:
        ts_features = pd.concat([ts_features, phases_features[phase]], axis=1)
    ts_features.dropna(inplace=True)
    exclusion_pid = exclusion_pid[np.in1d(exclusion_pid, ts_features.index)]
    ts_features = ts_features.drop(exclusion_pid, axis=0)
    ts_features.to_csv('type_' + str(i) + '_test.csv', index='process_id')
    print(inclusion_phase, ts_features.shape)

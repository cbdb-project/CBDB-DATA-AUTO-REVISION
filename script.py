import numpy as np
import pandas as pd
import opencc
from tqdm import tqdm
from glob import glob
import os
import os.path as osp


def main_7_1():
    def convert_trad_to_simp():
        # 初始化OpenCC的转换器，繁体(traditional)转(2)简体(simple)
        converter = opencc.OpenCC('t2s.json')

        # 读取Character Dictionary.xlsx文件
        char_dict = pd.read_excel("./Character Dictionary_20230627.xlsx", encoding='utf-8')

        # 为简繁转换创建字典
        trad_to_simp_dict = {}
        columns = char_dict.columns.values
        for index, row in char_dict.iterrows():
            simp = row["simp"]
            for col in columns:
                trad = row[col]
                if not pd.isna(trad):
                    trad_to_simp_dict[trad] = simp
                    # print(trad, simp)

        # 读取ALTNAME_DATA.csv文件
        altname_data = pd.read_csv("./ALTNAME_DATA.csv", low_memory=False)

        # 创建新列并使用简繁字典进行转换
        def _convert_trad_to_simp(text):
            if isinstance(text, float):
                # print(text)
                # exit(12)
                return ''
            return "".join(trad_to_simp_dict.get(char, converter.convert(char)) for char in text)

        altname_data["c_alt_name_chn_f"] = altname_data["c_alt_name_chn"].apply(_convert_trad_to_simp)

        # 将结果保存为新的CSV文件
        altname_data.to_csv("ALTNAME_DATA_converted.csv", index=False, encoding='utf-8')
        print('Done.')

    def check_conflict_name():
        # 读取繁简转换之后的文件
        data = pd.read_csv("ALTNAME_DATA_converted.csv", low_memory=False)
        personid, alt_name, alt_name_chn = data['c_personid'], data['c_alt_name'], data['c_alt_name_chn']
        num = data.shape[0]

        # # chech ascending personid
        # for i in range(num - 1):
        #     assert personid[i] <= personid[i+1], '{}, {}'.format(personid[i], personid[i+1])

        error_type = [''] * num

        def _add_err(curr, e):
            if curr == '':
                return e
            if e not in curr:
                curr = curr + ',' + e
            return curr

        for i in tqdm(range(num)):
            if error_type[i] != '':
                continue

            for j in range(i+1, num):
                if personid[i] == personid[j]:
                    if alt_name[i] != alt_name[j]:
                        # Inconsistent: personid相同, name拼音不同
                        error_type[i] = _add_err(error_type[i], 'Inconsistent')
                        error_type[j] = _add_err(error_type[j], 'Inconsistent')
                    elif alt_name_chn[i] != alt_name_chn[j]:
                        # Confounded: personid相同, name汉字繁简不同
                        error_type[i] = _add_err(error_type[i], 'Confounded')
                        error_type[j] = _add_err(error_type[j], 'Confounded')
                    else:
                        # Duplicated: personid相同, name拼音汉字相同, 但重复
                        error_type[i] = _add_err(error_type[i], 'Duplicated')
                        error_type[j] = _add_err(error_type[j], 'Duplicated')
                else:
                    break

        data['error_type'] = error_type
        data.to_csv("ALTNAME_DATA_verified.csv", index=False, encoding='utf-8')

    convert_trad_to_simp()
    check_conflict_name()


def main_7_7():
    save_dir = 'results/7.7'

    # altname_data = pd.read_csv("./ALTNAME_DATA/ALTNAME_DATA.csv", low_memory=False)
    # c_personid = altname_data['c_personid']
    # c_alt_name = altname_data['c_alt_name']
    # c_alt_name_chn = altname_data['c_alt_name_chn']
    # c_alt_name_type_code = altname_data['c_alt_name_type_code']
    # c_source = altname_data['c_source']
    # c_pages = altname_data['c_pages']
    #
    # num = altname_data.shape[0]
    #
    # ######### task 1
    # # a-z, A-Z, 0-9, 半角/全角空格, 框□
    # anomaly_list = list(range(97, 122+1)) + list(range(65, 90+1)) + list(range(48, 57+1)) + [32, 12288, 9633]
    # # for i in [12820, 12821]:
    # #     for j in [2, 3]:
    # #         char = altname_data.iloc[i, j]
    # #         print(i, j, char, type(char), len(char), ord(char))
    #
    # anomaly = []
    # for i in tqdm(range(num), desc='task1'):
    #     alt_name, alt_name_chn = altname_data['c_alt_name'][i], altname_data['c_alt_name_chn'][i]
    #     #  or any(c in anomaly_list for c in alt_name)
    #     flag = isinstance(alt_name, float) or isinstance(alt_name_chn, float) \
    #            or any(c in anomaly_list for c in alt_name_chn)
    #     anomaly.append(str(flag))
    #
    # altname_data['alt_name_anomaly'] = anomaly
    #
    # ######### task 2
    # converter = opencc.OpenCC('t2s')
    # sim2trad_dict = {}
    # for i in range(0x4e00, 0x9fd5+1):
    #     char_src = chr(i)
    #     char_dst = converter.convert(char_src)
    #     if char_src != char_dst:
    #         if char_dst not in sim2trad_dict:
    #             sim2trad_dict[char_dst] = []
    #         sim2trad_dict[char_dst].append(char_src)
    # sim2trad_list = []
    #
    # max_len = max(len(v) for v in sim2trad_dict.values())
    # sim_list = []
    # for sim, trad_list in sim2trad_dict.items():
    #     sim_list.append(sim)
    #     placeholder = [''] * (max_len - len(trad_list))
    #     sim2trad_list.append([sim]+trad_list+placeholder)
    # df = pd.DataFrame(data=sim2trad_list, columns=['sim'] + [f'trad{i+1}' for i in range(max_len)], index=sim_list)
    # df.to_csv(f'{save_dir}/sim2trad_dict_opencc.csv', index=False, encoding='utf-8')
    # df.to_excel(f'{save_dir}/sim2trad_dict_opencc.xlsx', index=False, encoding='utf-8')
    #
    # alt_name_chn_simplified = []
    # alt_name_chn_sim = []
    # char_dict = pd.read_excel("./ALTNAME_DATA/Character Dictionary_20230627.xlsx")
    # # 为简繁转换创建字典
    # trad_to_simp_dict = {}
    # columns = char_dict.columns.values
    # for index, row in char_dict.iterrows():
    #     simp = row["simp"]
    #     for col in columns:
    #         trad = row[col]
    #         if not pd.isna(trad):
    #             trad_to_simp_dict[trad] = simp
    # for altname_chn in tqdm(c_alt_name_chn, desc='task2'):
    #     if isinstance(altname_chn, float):
    #         flag = None
    #         chn_sim = altname_chn
    #     else:
    #         chn_sim = "".join(trad_to_simp_dict.get(char, converter.convert(char)) for char in altname_chn)
    #         flag = chn_sim == altname_chn
    #     alt_name_chn_simplified.append(str(flag))
    #     alt_name_chn_sim.append(chn_sim)
    # altname_data['alt_name_chn_sim'] = alt_name_chn_sim
    # altname_data['alt_name_alt_name_chn_simplified'] = alt_name_chn_simplified
    #
    # ######### task 4
    # alt_name_error_type = []
    # for i in tqdm(range(num), desc='task4'):
    #     # for j in range(i+1, num):
    #     #     if c_personid[i] != c_personid[j]:
    #     #         break
    #     if c_alt_name_type_code[i] == 0:
    #         alt_name_error_type.append('Delete')
    #     elif c_alt_name_type_code[i] == 4:
    #         alt_name_error_type.append('Check')
    #     else:
    #         alt_name_error_type.append('')
    # altname_data['alt_name_error_type'] = alt_name_error_type
    #
    # ######### task 5
    # source_pages_error_type = []
    # error_dict = {-1: 'Delete', 0: 'Others', 1: '', 2: 'Supplement'}
    # i = 0
    # while len(source_pages_error_type) < num:
    #     # print(f'task5: {i}/{num}, {len(source_pages_error_type)}')
    #     same_person = [[c_source[i], c_pages[i]]]
    #     for j in range(i+1, num):
    #         # or alt_name_chn_simplified[i] != alt_name_chn_simplified[j]
    #         if c_personid[i] != c_personid[j] or c_alt_name[i] != c_alt_name[j]:
    #             break
    #         same_person.append([c_source[j], c_pages[j]])
    #     is_complete = [int(all(x != '' for x in person)) for person in same_person]
    #     if any(is_complete):
    #         first_complete = np.argmax(np.array(is_complete))
    #         for k in range(len(is_complete)):
    #             if not is_complete[k]:
    #                 if all(x == '' for x in same_person[k]):
    #                     is_complete[k] = -1  # delete
    #             elif k != first_complete:
    #                 is_complete[k] = 0  # conflict
    #     else:
    #         for k in range(len(is_complete)):
    #             if all(x == '' for x in same_person[k]):
    #                 is_complete[k] = -1  # delete
    #
    #     for flag in is_complete:
    #         source_pages_error_type.append(error_dict[flag])
    #
    #     i = j
    #
    # altname_data.to_csv(f"{save_dir}/ALTNAME_DATA_results.csv", index=False, encoding='utf-8')

    ######### task 3
    save_dir = f'{save_dir}/CBDB_tables_simplyfied'
    os.makedirs(save_dir, exist_ok=True)
    invalid_colmns = ['c_notes', 'c_alt_name_chn', 'c_name_chn', 'c_title_chn']

    converter = opencc.OpenCC('t2s')
    trad_to_simp_dict = {}
    char_dict = pd.read_excel("./ALTNAME_DATA/Character Dictionary_20230627.xlsx")
    columns = char_dict.columns.values
    for index, row in char_dict.iterrows():
        simp = row["simp"]
        for col in columns:
            trad = row[col]
            if not pd.isna(trad):
                trad_to_simp_dict[trad] = simp

    def _convert_trad_to_simp(text):
        if isinstance(text, float):
            return text
        return "".join(trad_to_simp_dict.get(char, converter.convert(char)) for char in text)

    for file_path in tqdm(glob('./CBDB_tables/*.csv'), desc='task3'):
        data = pd.read_csv(file_path, low_memory=False)
        for col in data.columns.values:
            if col in invalid_colmns:
                continue
            is_chn = False
            for text in data[col][:100]:
                if isinstance(text, str) and any(0x4e00 <= ord(c) <= 0x9fd5 for c in text):
                    is_chn = True
            if is_chn:
                data[col] = data[col].apply(_convert_trad_to_simp)
        data.to_csv(f"{save_dir}/{osp.basename(file_path)}", index=False, encoding='utf-8')


def main_7_12():
    save_dir = 'results/7.12'
    os.makedirs(save_dir, exist_ok=True)

    altname_data = pd.read_csv("./ALTNAME_DATA/ALTNAME_DATA.csv", low_memory=False)
    c_personid = altname_data['c_personid']
    c_alt_name, c_alt_name_chn = altname_data['c_alt_name'], altname_data['c_alt_name_chn']
    c_alt_name_type_code = altname_data['c_alt_name_type_code']
    c_source, c_pages = altname_data['c_source'], altname_data['c_pages']
    c_secondary_source_author, c_notes = altname_data['c_secondary_source_author'], altname_data['c_notes']

    num = altname_data.shape[0]

    ######### task 1
    # a-z, A-Z, 0-9, 半角/全角空格, 框□
    anomaly_list = list(range(97, 122+1)) + list(range(65, 90+1)) + list(range(48, 57+1)) + [32, 12288, 9633]
    # a-z, A-Z, 半角空格, '分字符
    altname_list = list(range(97, 122+1)) + list(range(65, 90+1)) + [32, 39]
    # altname_chn_list = list(range(0x2E80, 0x2FDF+1)) + list(range(0x3400, 0x4DBF+1)) + list(range(0x4E00, 0x9FFF+1))

    columns = altname_data.columns.values
    anomaly_altname_df = pd.DataFrame(columns=columns)
    anomaly_altname_chn_df = pd.DataFrame(columns=columns)
    for i in tqdm(range(num), desc='task1'):
        alt_name, alt_name_chn = altname_data['c_alt_name'][i], altname_data['c_alt_name_chn'][i]
        if isinstance(alt_name, float) or any(ord(c) not in altname_list for c in alt_name):
            anomaly_altname_df.loc[len(anomaly_altname_df.index)] = altname_data.iloc[i]
        # if isinstance(alt_name_chn, float) or any(ord(c) not in altname_chn_list for c in alt_name_chn):
        #     anomaly_altname_chn_df.loc[len(anomaly_altname_chn_df.index)] = altname_data.iloc[i]
        if isinstance(alt_name_chn, float) or any(ord(c) in anomaly_list for c in alt_name_chn):
            anomaly_altname_chn_df.loc[len(anomaly_altname_chn_df.index)] = altname_data.iloc[i]

    anomaly_altname_df.to_csv(f"{save_dir}/1_ALTNAME_DATA_anomaly_altname.csv", index=False, encoding='utf-8')
    anomaly_altname_chn_df.to_csv(f"{save_dir}/1_ALTNAME_DATA_anomaly_altname_chn.csv", index=False, encoding='utf-8')

    ######### task 2
    sim2trad_dict = {}
    with open(r'F:\pycharm_projects\venv\lib\site-packages\opencc\dictionary\TSCharacters.txt',
              'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            words = line.split()[:2]
            # if len(words) > 2:
            #     print(line, words)
            #     continue
            trad, sim = words
            if sim not in sim2trad_dict:
                sim2trad_dict[sim] = []
            sim2trad_dict[sim].append(trad)

    sim2trad_list = []
    max_len = max(len(v) for v in sim2trad_dict.values())
    sim_list = []
    for sim, trad_list in sim2trad_dict.items():
        sim_list.append(sim)
        placeholder = [''] * (max_len - len(trad_list))
        sim2trad_list.append([sim]+trad_list+placeholder)
    df = pd.DataFrame(data=sim2trad_list, columns=['sim'] + [f'trad{i+1}' for i in range(max_len)], index=sim_list)
    df.to_csv(f'{save_dir}/2_sim2trad_dict_opencc.csv', index=False, encoding='utf-8')

    alt_name_chn_simplified = []
    alt_name_chn_sim = []
    char_dict = pd.read_excel("./ALTNAME_DATA/Character Dictionary_20230627.xlsx")
    # 为简繁转换创建字典
    trad_to_simp_dict = {}
    columns = char_dict.columns.values
    for index, row in char_dict.iterrows():
        simp = row["simp"]
        for col in columns:
            trad = row[col]
            if not pd.isna(trad):
                trad_to_simp_dict[trad] = simp
    # opencc简繁转换器
    converter = opencc.OpenCC('t2s')
    # 开始简繁转换
    for altname_chn in tqdm(c_alt_name_chn, desc='task2'):
        if isinstance(altname_chn, float):
            flag = None
            chn_sim = altname_chn
        else:
            chn_sim = "".join(trad_to_simp_dict.get(char, converter.convert(char)) for char in altname_chn)
            flag = chn_sim == altname_chn
        alt_name_chn_simplified.append(str(flag))
        alt_name_chn_sim.append(chn_sim)
    altname_data['alt_name_chn_sim'] = alt_name_chn_sim
    altname_data['alt_name_alt_name_chn_simplified'] = alt_name_chn_simplified

    ######### task 4
    alt_name_error_type = []
    i = 0
    while i < num and len(alt_name_error_type) < num:
        same_person = [c_alt_name_type_code[i]]
        for j in range(i+1, num):
            if c_personid[i] != c_personid[j] or c_alt_name[i] != c_alt_name[j] \
                    or c_alt_name_chn[i] != c_alt_name_chn[j]:
                break
            same_person.append(c_alt_name_type_code[j])

        cur_num = len(same_person)
        if cur_num == 1:
            # 只有一个人，空
            alt_name_error_type.append('')
        elif all(code in [0, 4] for code in same_person) and all(code in same_person for code in [0, 4]):
            # delete: 有且仅有0、4两种情况
            alt_name_error_type.extend(['Delete'] * cur_num)
        else:
            # check: 其他所有情况都要人工检查
            alt_name_error_type.extend(['Check'] * cur_num)

        i = j

    altname_data['alt_name_error_type'] = alt_name_error_type

    ######### task 5
    source_pages_error_type = []
    i = 0
    while i < num and len(source_pages_error_type) < num:
        # print(f'task5: {i}/{num}, {len(source_pages_error_type)}')
        same_person = [[c_source[i], c_pages[i], c_secondary_source_author[i], c_notes[i]]]
        for j in range(i+1, num):
            if c_personid[i] != c_personid[j] or c_alt_name[i] != c_alt_name[j] \
                    or c_alt_name_chn[i] != c_alt_name_chn[j]:
                break
            same_person.append([c_source[j], c_pages[j], c_secondary_source_author[j], c_notes[j]])

        cur_num = len(same_person)
        if cur_num == 1:
            # 只有一个人，空
            source_pages_error_type.append('')
        elif cur_num == 2:
            valid = np.array([[x == '' for x in person] for person in same_person])  # shape(2,4)
            k1, k2 = np.argsort(-valid[:, 0].astype(np.int32))  # 按照有/无source排序
            # Delete: 一行有source也有pages，另一行没有source也没有pages
            if all(valid[k1][:2]) and not any(valid[k2][:2]):
                source_pages_error_type.extend(['Delete'] * cur_num)
            # Delete: 一行有source和后面的信息但没有pages，另一行只有pages
            elif (valid[k1][0] and not valid[k1][1] and any(valid[k1][2:])) and \
                    (not valid[k2][0] and valid[k2][1] and not any(valid[k2][2:])):
                source_pages_error_type.extend(['Delete'] * cur_num)
            # Supplement: 两行都有source，但一行有pages和后面的，一行没有
            elif all(valid[:, 0]) and \
                    ((valid[k1][1] and any(valid[k1][2:]) and not any(valid[k2][1:])) or
                     (valid[k2][1] and any(valid[k2][2:]) and not any(valid[k1][1:]))):
                source_pages_error_type.extend(['Supplement'] * cur_num)
            # Merge: 一行有source但没有pages和后面的，另一行没有source但有pages和后面的
            elif (valid[k1][0] and not any(valid[k1][1:])) and \
                 (not valid[k2][0] and valid[k2][1] and valid[k2][2:]):
                source_pages_error_type.extend(['Merge'] * cur_num)
            # 其他没考虑到的情况，都需要人工check
            else:
                source_pages_error_type.extend(['Check'] * cur_num)
        else:
            # 有大于等于3个人的情况，需要人工check
            source_pages_error_type.extend(['Check'] * cur_num)

        i = j

    altname_data['source_pages_error_type'] = source_pages_error_type

    altname_data.to_csv(f"{save_dir}/0_ALTNAME_DATA_results.csv", index=False, encoding='utf-8')

    ######### task 8
    dynasty_info_data = pd.read_csv("./CBDB_TABLES/DYNASTIES.csv", low_memory=False)
    c_dy, c_dynasty = dynasty_info_data['c_dy'], list(dynasty_info_data['c_dynasty'])
    dynasty_types = ['Tang', 'Song', 'Liao', 'Jin', 'Yuan', 'Ming', 'Qing', 'Republic of China']
    dynasty_index = [c_dy[c_dynasty.index(dy)] for dy in dynasty_types]

    biog_main_data = pd.read_csv("./CBDB_TABLES/BIOG_MAIN.csv", low_memory=False)
    c_dy = np.array(biog_main_data['c_dy']).astype(np.int32)

    count_dict = {dy: 0 for dy in dynasty_index}
    for dy in c_dy:
        if np.isnan(dy):
            continue
        if dy in dynasty_index:
            count_dict[dy] += 1
    df = pd.DataFrame(columns=dynasty_types)
    df.loc[len(df.index)] = list(count_dict.values())
    df.to_csv(f"{save_dir}/8_1_dynasty_statistic.csv", index=False, encoding='utf-8')

    data_type_dict = {
        'Number of Persons': 'BIOG_MAIN', 'Social Associations': 'ASSOC_DATA',
        'Biographical Addresses': 'BIOG_ADDR_DATA', 'Alternate Names': 'ALTNAME_DATA',
        'Kin Relationships': 'KIN_DATA', 'Entry into Office': 'ENTRY_DATA',
        'Office Postings': 'POSTED_TO_OFFICE_DATA', 'Social Distinction': 'STATUS_DATA',
        'Texts': 'TEXT_CODES'
    }
    count_dict = {k: 0 for k in data_type_dict.keys()}

    for data_type, table_name in data_type_dict.items():
        data_df = pd.read_csv(f"./CBDB_TABLES/{table_name}.csv", low_memory=False)
        count_dict[data_type] = data_df.shape[0]
    df = pd.DataFrame(columns=list(count_dict.keys()))
    df.loc[len(df.index)] = list(count_dict.values())
    df.to_csv(f"{save_dir}/8_2_datatype_statistic.csv", index=False, encoding='utf-8')


def main_7_19():
    save_dir = 'results/7.19'
    os.makedirs(save_dir, exist_ok=True)

    ################ task 2
    altname_data = pd.read_csv("./results/7.12/1_ALTNAME_DATA_anomaly_altname_chn.csv", low_memory=False)
    # C, D, I
    c_alt_name, c_alt_name_chn, c_notes = \
        altname_data['c_alt_name'], altname_data['c_alt_name_chn'], altname_data['c_notes']
    error_types = []

    def _invalid(x):
        return not isinstance(x, str) or all(c in ['', '□', '[n/a]'] for c in x)

    def _exist(x, tag):
        return isinstance(x, str) and tag in x

    def _is_all(x, xrange):
        return isinstance(x, str) and all(ord(c) in xrange for c in x)

    def _is_not_all(x, xrange):
        return isinstance(x, str) and all(ord(c) not in xrange for c in x)

    alphabet_range = list(range(97, 122+1)) + list(range(65, 90+1))
    for alt_name, alt_name_chn, notes in zip(c_alt_name, c_alt_name_chn, c_notes):
        status_list = []

        if _invalid(alt_name) and _invalid(alt_name_chn) and _invalid(notes):
            status_list.append('Delete')
        else:
            if _exist(alt_name, '□') or _exist(alt_name_chn, '□'):
                status_list.append('Somebody')
            if _is_all(alt_name_chn, alphabet_range + [32]) and _is_all(alt_name, alphabet_range + [32]):
                status_list.append('Duplicate')
            elif _is_all(alt_name_chn, alphabet_range + [32]) and _is_not_all(alt_name, alphabet_range):
                status_list.append('Interconvert')
                alt_name, alt_name_chn = alt_name_chn, alt_name
            if (_is_all(alt_name, alphabet_range + [32]) and _exist(alt_name, '  ')) or \
                    (_is_not_all(alt_name_chn, alphabet_range) and _exist(alt_name_chn, ' ')):
                status_list.append('Blank')

        error_types.append(','.join(status_list))
    altname_data['error_types'] = error_types
    altname_data.to_csv(f"{save_dir}/2_ALTNAME_DATA_err_types.csv", index=False, encoding='utf-8')

    ################ task 3
    altname_data = pd.read_csv("./CBDB_TABLES/ALTNAME_DATA.csv", low_memory=False)
    c_alt_name_chn = altname_data['c_alt_name_chn']
    sim2trad_data = pd.read_csv("./results/7.12/2_sim2trad_dict_opencc.csv", low_memory=False)
    sim_data = list(sim2trad_data['sim'])
    flags = [str(isinstance(alt_name_chn, str) and any(c in sim_data for c in alt_name_chn))
             for alt_name_chn in tqdm(c_alt_name_chn)]
    altname_data['sim_exists'] = flags
    altname_data.to_csv(f"{save_dir}/3_ALTNAME_DATA_sim_exists.csv", index=False, encoding='utf-8')

    ################ task 5
    kin_data = pd.read_csv("./CBDB_TABLES/KIN_DATA.csv", low_memory=False)
    c_personid, c_kin_id, c_kin_code = \
        kin_data['c_personid'], kin_data['c_kin_id'], kin_data['c_kin_code']

    biog_data = pd.read_csv("./CBDB_TABLES/BIOG_MAIN.csv", low_memory=False)
    person2data = {k: [v1, v2] for k, v1, v2 in
                   zip(biog_data['c_personid'], biog_data['c_name_chn'], biog_data['c_female'])}

    kinship_data = pd.read_csv("./CBDB_TABLES/KINSHIP_CODES.csv", low_memory=False)
    kin2data = {k: [v1, v2] for k, v1, v2 in
                zip(kinship_data['c_kincode'], kinship_data['c_kinrel_chn'], kinship_data['c_kinrel'])}

    c_personid_name_chn = list(person2data[k][0] for k in c_personid)
    c_kin_id_name_chn, c_kin_id_female = tuple(zip(*list(person2data[k] for k in c_kin_id)))
    c_kinrel_chn, c_kinrel = tuple(zip(*list(kin2data[k] for k in c_kin_code)))

    data_dict = {'c_personid': c_personid, 'c_personid_name_chn': c_personid_name_chn,
                 'c_kin_id': c_kin_id, 'c_kin_id_name_chn': c_kin_id_name_chn, 'c_kin_code': c_kin_code,
                 'c_kin_id_female': c_kin_id_female, 'c_kinrel_chn': c_kinrel_chn, 'c_kinrel': c_kinrel}
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{save_dir}/5_KIN_DATA_NEW.csv", index=False, encoding='utf-8')

    kin_code_stat = {}
    for kin_code, kinrel_chn in zip(c_kin_code, c_kinrel_chn):
        if kin_code not in kin_code_stat:
            kin_code_stat[kin_code] = {'kinrel_chn': kinrel_chn, 'num': 1}
        else:
            kin_code_stat[kin_code]['num'] += 1
    data_dict = {'c_kin_code': list(kin_code_stat.keys()),
                 'c_kinrel_chn': list(x['kinrel_chn'] for x in kin_code_stat.values()),
                 'c_kin_code_num': list(x['num'] for x in kin_code_stat.values())}
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{save_dir}/5_KIN_DATA_NEW_kin_code_stat.csv", index=False, encoding='utf-8')

    kin_female_stat = {}
    for kin_id_female in c_kin_id_female:
        if kin_id_female not in kin_female_stat:
            kin_female_stat[kin_id_female] = 1
        else:
            kin_female_stat[kin_id_female] += 1
    data_dict = {'c_kin_id_female': list(kin_female_stat.keys()),
                 'c_kin_id_female_num': list(kin_female_stat.values())}
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{save_dir}/5_KIN_DATA_NEW_kin_female_stat.csv", index=False, encoding='utf-8')


def main_7_26():
    save_dir = 'results/7.26'
    os.makedirs(save_dir, exist_ok=True)

    ################ task 2
    sim2trad_dict = {}
    with open(r'F:\pycharm_projects\venv\lib\site-packages\opencc\dictionary\TSCharacters.txt',
              'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            words = line.split()[:2]
            # if len(words) > 2:
            #     print(line, words)
            #     continue
            trad, sim = words
            if trad == sim:
                continue
            if sim not in sim2trad_dict:
                sim2trad_dict[sim] = []
            sim2trad_dict[sim].append(trad)

    sim2trad_list = []
    max_len = max(len(v) for v in sim2trad_dict.values())
    sim_list = []
    for sim, trad_list in sim2trad_dict.items():
        sim_list.append(sim)
        placeholder = [''] * (max_len - len(trad_list))
        sim2trad_list.append([sim] + trad_list + placeholder)
    df = pd.DataFrame(data=sim2trad_list, columns=['sim'] + [f'trad{i+1}' for i in range(max_len)], index=sim_list)
    df.to_csv(f'{save_dir}/2_sim2trad_dict_opencc.csv', index=False, encoding='utf-8')

    ################ task 3
    kin_data = pd.read_csv("./results/7.19/5_KIN_DATA_NEW.csv", low_memory=False)
    c_personid, c_kin_id, c_kin_code, c_kinrel_chn, c_kin_id_female = \
        np.array(kin_data['c_personid']), np.array(kin_data['c_kin_id']), np.array(kin_data['c_kin_code']), \
        kin_data['c_kinrel_chn'], kin_data['c_kin_id_female']

    kin_code_stat = {}
    for kin_code, kinrel_chn, kin_id_female in zip(c_kin_code, c_kinrel_chn, c_kin_id_female):
        if kin_code not in kin_code_stat:
            kin_code_stat[kin_code] = {'kinrel_chn': kinrel_chn, 'females': []}
        else:
            kin_code_stat[kin_code]['females'].append(kin_id_female)
    data_dict = {'c_kin_code': [], 'c_kinrel_chn': [], 'c_kin_code_num': [],
                 'c_kin_code_num_female': [], 'c_kin_code_num_male': []}
    for k, v in kin_code_stat.items():
        data_dict['c_kin_code'].append(k)
        data_dict['c_kinrel_chn'].append(v['kinrel_chn'])
        code_num = len(v['females'])
        code_num_female = int(sum(v['females']))
        data_dict['c_kin_code_num'].append(code_num)
        data_dict['c_kin_code_num_female'].append(code_num_female)
        data_dict['c_kin_code_num_male'].append(code_num-code_num_female)

    df = pd.DataFrame(data_dict)
    df.to_csv(f"{save_dir}/3_KIN_DATA_NEW_kin_female_stat.csv", index=False, encoding='utf-8')

    kinship_codes = pd.read_csv("./CBDB_TABLES/KINSHIP_CODES.csv", low_memory=False)
    c_kincode, c_kin_pair1, c_kin_pair2 = \
        kinship_codes['c_kincode'], kinship_codes['c_kin_pair1'], kinship_codes['c_kin_pair2']
    kincode_pair_dict = {k: np.array([p1, p2]) for k, p1, p2 in zip(c_kincode, c_kin_pair1, c_kin_pair2)}

    kin_data_num = kin_data.shape[0]
    xrange = np.arange(kin_data_num)
    kin_error_type = ['' for _ in xrange]
    valid = np.full((kin_data_num,), False)
    for i, (personid, kin_id, kin_code) in \
            tqdm(enumerate(zip(c_personid, c_kin_id, c_kin_code)), total=kin_data_num):
        if valid[i]:
            continue
        kin_person_indices = np.argwhere(personid == kin_id)
        kin_person_kin_ids = c_kin_id[kin_person_indices]
        kin_person_kin_codes = c_kin_code[kin_person_indices]
        paired_codes = kincode_pair_dict[kin_code]
        same_kinship = kin_person_kin_ids == personid
        if same_kinship.sum() == 0:
            kin_error_type[i] = 'Missing'
        else:
            valid_pairs = (kin_person_kin_codes.reshape((-1, 1)) == paired_codes.reshape((1, -1))).any(axis=1)
            for j in xrange[kin_person_indices][valid_pairs]:
                valid[j] = True
            for j in xrange[kin_person_indices][~valid_pairs]:
                kin_error_type[j] = 'Contradictory'
                valid[j] = True
            if (~valid_pairs).all():
                kin_error_type[i] = 'Contradictory'

    kin_data['error_type'] = kin_error_type
    kin_data.to_csv(f"{save_dir}/3_KIN_DATA_NEW_error_type.csv", index=False, encoding='utf-8')

    ################ task 4
    entry_data = pd.read_csv("./CBDB_TABLES/ENTRY_DATA.csv", low_memory=False)
    c_entry_code, c_pages, c_notes, c_age, c_entry_nh_year, c_nianhao_id, c_entry_code, c_source = \
        entry_data['c_entry_code'], entry_data['c_pages'], entry_data['c_notes'], entry_data['c_age'], \
        entry_data['c_entry_nh_year'], entry_data['c_nianhao_id'], entry_data['c_entry_code'], entry_data['c_source']

    NIAN_HAO = pd.read_csv("./CBDB_TABLES/NIAN_HAO.csv", low_memory=False)
    c_nianhao_id_total = NIAN_HAO['c_nianhao_id']
    ENTRY_CODES = pd.read_csv("./CBDB_TABLES/ENTRY_CODES.csv", low_memory=False)
    c_entry_code_total = ENTRY_CODES['c_entry_code']
    TEXT_CODES = pd.read_csv("./CBDB_TABLES/TEXT_CODES.csv", low_memory=False)
    c_source_total = TEXT_CODES['c_source']

    def _empty(x):
        return not x or (isinstance(x, float) and np.isnan(x))

    entry_error_type = []
    for entry_code, pages, notes, age, entry_nh_year, nianhao_id, entry_code, source in \
            zip(c_entry_code, c_pages, c_notes, c_age, c_entry_nh_year, c_nianhao_id, c_entry_code, c_source):
        error_type = []
        if _empty(entry_code) and _empty(pages) and _empty(notes):
            error_type.append('EmptyEntry')
        if age < 0 or age > 100:
            error_type.append('WrongAge')
        if entry_nh_year >= 61:
            error_type.append('HugeNHYear')
        if nianhao_id not in c_nianhao_id_total:
            error_type.append('NoNianHao')
        if entry_code not in c_entry_code_total:
            error_type.append('NoEntryCode')
        if source not in c_source_total:
            error_type.append('NoSource')
        if not _empty(source) and _empty(pages):
            error_type.append('NoPage')
        entry_error_type.append(','.join(error_type))

    entry_data['error_type'] = entry_error_type
    entry_data.to_csv(f"{save_dir}/4_ENTRY_DATA_error.csv", index=False, encoding='utf-8')


def main_8_2():
    save_dir = 'results/8.2'
    os.makedirs(save_dir, exist_ok=True)

    # ################ task 1-4
    # kin_data_female = pd.read_excel("./results/7.26/3_KIN_DATA_NEW_kin_female_stat_new.xlsx")
    # c_kin_code, c_kinrel_chn, c_kin_female_ratio, c_kinrel_female_true = \
    #     np.array(kin_data_female['c_kin_code']), np.array(kin_data_female['c_kinrel_chn']), \
    #     np.array(kin_data_female['c_kin_female_ratio']), np.array(kin_data_female['c_kinrel_female_true'])
    # kin_dict = {code: [chn, ratio, female] for (code, chn, ratio, female) in \
    #             zip(c_kin_code, c_kinrel_chn, c_kin_female_ratio, c_kinrel_female_true)}
    #
    # kin_data = pd.read_csv("./results/7.19/5_KIN_DATA_NEW.csv", low_memory=False)
    # c_kin_code, c_kin_id_female = np.array(kin_data['c_kin_code']), kin_data['c_kin_id_female']
    #
    # kin_check = []
    # kin_female_stat, c_kinrel_female = [], []
    # for kin_code, kin_id_female in zip(c_kin_code, c_kin_id_female):
    #     chn, ratio, female = kin_dict[kin_code]
    #     kin_female_stat.append(ratio)
    #     c_kinrel_female.append(female)
    #     kin_check.append(kin_id_female != female)
    # kin_data['kin_female_stat'] = kin_female_stat
    # kin_data['c_kinrel_female'] = c_kinrel_female
    # kin_data_to_check = kin_data.iloc[kin_check, :]
    # kin_data_to_check.to_csv(f"{save_dir}/1_4_KIN_DATA_NEW_CHECK.csv", index=False, encoding='utf-8')

    ################ task 1-5
    kin_data = pd.read_csv("./results/7.19/5_KIN_DATA_NEW.csv", low_memory=False)
    c_personid, c_kin_id, c_kin_code, c_kinrel_chn, c_kin_id_female = \
        np.array(kin_data['c_personid']), np.array(kin_data['c_kin_id']), np.array(kin_data['c_kin_code']), \
        np.array(kin_data['c_kinrel_chn']), np.array(kin_data['c_kin_id_female'])

    kinship_codes = pd.read_csv("./CBDB_TABLES/KINSHIP_CODES.csv", low_memory=False)
    c_kincode, c_kin_pair1, c_kin_pair2 = \
        kinship_codes['c_kincode'], kinship_codes['c_kin_pair1'], kinship_codes['c_kin_pair2']
    kincode_pair_dict = {k: np.array([p1, p2]) for k, p1, p2 in zip(c_kincode, c_kin_pair1, c_kin_pair2)}

    kin_data_num = kin_data.shape[0]
    xrange = np.arange(kin_data_num)
    kin_error_type = ['' for _ in xrange]
    valid = np.full((kin_data_num,), False)
    for i, (personid, kin_id, kin_code) in \
            tqdm(enumerate(zip(c_personid, c_kin_id, c_kin_code)), total=kin_data_num):
        if valid[i]:
            continue
        kin_person_indices = np.argwhere(c_personid == kin_id)
        if len(kin_person_indices) == 0:
            kin_error_type[i] = 'Missing'
            continue
        kin_person_indices = kin_person_indices[:, 0]
        kin_person_kin_ids = c_kin_id[kin_person_indices]
        kin_person_kin_codes = c_kin_code[kin_person_indices]
        paired_codes = kincode_pair_dict[kin_code]
        same_kinship = kin_person_kin_ids == personid
        if same_kinship.sum() == 0:
            kin_error_type[i] = 'Missing'
        else:
            valid_pairs = (kin_person_kin_codes.reshape((-1, 1)) == paired_codes.reshape((1, -1))).any(axis=1)
            for j in xrange[kin_person_indices][same_kinship & valid_pairs]:
                valid[j] = True
            for j in xrange[kin_person_indices][same_kinship & ~valid_pairs]:
                kin_error_type[j] = 'Contradictory'
                valid[j] = True
            if (~valid_pairs).all():
                kin_error_type[i] = 'Contradictory'

    kin_data['error_type'] = kin_error_type
    kin_data.to_csv(f"{save_dir}/1_5_KIN_DATA_NEW_error_type.csv", index=False, encoding='utf-8')

    # ################ task 2
    # entry_data = pd.read_csv("./CBDB_TABLES/ENTRY_DATA.csv", low_memory=False)
    # c_entry_code, c_pages, c_notes, c_age, c_entry_nh_year, c_year, c_nianhao_id, c_entry_code, c_source = \
    #     entry_data['c_entry_code'], entry_data['c_pages'], entry_data['c_notes'], entry_data['c_age'], \
    #     entry_data['c_entry_nh_year'], entry_data['c_year'], entry_data['c_nianhao_id'], \
    #     entry_data['c_entry_code'], entry_data['c_source']
    #
    # NIAN_HAO = pd.read_csv("./CBDB_TABLES/NIAN_HAO.csv", low_memory=False)
    # c_nianhao_id_total = np.array(NIAN_HAO['c_nianhao_id'])
    # ENTRY_CODES = pd.read_csv("./CBDB_TABLES/ENTRY_CODES.csv", low_memory=False)
    # c_entry_code_total = np.array(ENTRY_CODES['c_entry_code'])
    # TEXT_CODES = pd.read_csv("./CBDB_TABLES/TEXT_CODES.csv", low_memory=False)
    # c_source_total = np.array(TEXT_CODES['c_textid'])
    #
    # def _empty(x):
    #     return not x or (isinstance(x, float) and np.isnan(x))
    #
    # entry_error_type = []
    # for entry_code, pages, notes, age, entry_nh_year, year, nianhao_id, entry_code, source in \
    #         zip(c_entry_code, c_pages, c_notes, c_age, c_entry_nh_year, c_year, c_nianhao_id, c_entry_code, c_source):
    #     error_type = []
    #     if _empty(entry_code) and _empty(pages) and _empty(notes):
    #         error_type.append('EmptyEntry')
    #     if age < 0 or age > 100:
    #         error_type.append('WrongAge')
    #     if entry_nh_year > 61:
    #         error_type.append('HugeNHYear')
    #     if not _empty(nianhao_id) and nianhao_id not in c_nianhao_id_total:
    #         error_type.append('NoNianHao')
    #     if not _empty(year) and np.isnan(nianhao_id):
    #         error_type.append('EmptyNianHao')
    #     if entry_code not in c_entry_code_total:
    #         error_type.append('NoEntryCode')
    #     if source not in c_source_total:
    #         error_type.append('NoSource')
    #     if not _empty(source) and _empty(pages):
    #         error_type.append('NoPage')
    #     if _empty(source) and not _empty(pages):
    #         error_type.append('NoSourceHavePage')
    #
    #     entry_error_type.append(','.join(error_type))
    #
    # entry_data['error_type'] = entry_error_type
    # entry_data.to_csv(f"{save_dir}/2_ENTRY_DATA_error.csv", index=False, encoding='utf-8')
    #
    # ################ task 3
    # dynasty_info_data = pd.read_csv("./CBDB_TABLES/DYNASTIES.csv", low_memory=False)
    # c_dy, c_dynasty = dynasty_info_data['c_dy'], list(dynasty_info_data['c_dynasty'])
    # dynasty_types = ['Tang', 'Song', 'Liao', 'Jin', 'Yuan', 'Ming', 'Qing', 'Republic of China']
    # dynasty_index = [c_dy[c_dynasty.index(dy)] for dy in dynasty_types]
    # wudai_index = np.array([7, 34, 47, 48, 52, 49, 36, 75, 9, 8, 11, 38, 12, 13, 55, 10, 66]).astype(np.int32)
    #
    # biog_main_data = pd.read_csv("./CBDB_TABLES/BIOG_MAIN.csv", low_memory=False)
    # c_dy = np.array(biog_main_data['c_dy']).astype(np.int32)
    #
    # count_dict = {dy: 0 for dy in dynasty_index}
    # for dy in c_dy:
    #     if np.isnan(dy):
    #         continue
    #     if dy in dynasty_index:
    #         count_dict[dy] += 1
    # count_dict[-1] = (c_dy.reshape((-1, 1)) == wudai_index.reshape((1, -1))).any(axis=1).sum()
    # df = pd.DataFrame(columns=dynasty_types + ['WuDai'])
    # df.loc[len(df.index)] = list(count_dict.values())
    # df.to_csv(f"{save_dir}/3_dynasty_statistic.csv", index=False, encoding='utf-8')


def main_8_11():
    save_dir = 'results/8.11'
    os.makedirs(save_dir, exist_ok=True)

    ################ task 1
    kin_data = pd.read_csv("./results/7.19/5_KIN_DATA_NEW.csv", low_memory=False)
    c_personid, c_kin_id, c_kin_code, c_kinrel_chn, c_kin_id_female = \
        np.array(kin_data['c_personid']), np.array(kin_data['c_kin_id']), np.array(kin_data['c_kin_code']), \
        np.array(kin_data['c_kinrel_chn']), np.array(kin_data['c_kin_id_female'])

    kinship_codes = pd.read_csv("./CBDB_TABLES/KINSHIP_CODES.csv", low_memory=False)
    c_kincode, c_kin_pair1, c_kin_pair2 = \
        kinship_codes['c_kincode'], kinship_codes['c_kin_pair1'], kinship_codes['c_kin_pair2']
    kincode_pair_dict = {k: [p1, p2] for k, p1, p2 in zip(c_kincode, c_kin_pair1, c_kin_pair2)}
    for k, p1, p2 in zip(c_kincode, c_kin_pair1, c_kin_pair2):
        if not np.isnan(p1) and k not in kincode_pair_dict[p1]:
            kincode_pair_dict[p1].append(k)
        if not np.isnan(p2) and k not in kincode_pair_dict[p2]:
            kincode_pair_dict[p2].append(k)
    kincode_pair_dict = {k: np.array(v) for k, v in kincode_pair_dict.items()}

    kin_data_num = kin_data.shape[0]
    xrange = np.arange(kin_data_num)
    kin_error_type = ['' for _ in xrange]
    valid = np.full((kin_data_num,), False)
    for i, (personid, kin_id, kin_code) in \
            tqdm(enumerate(zip(c_personid, c_kin_id, c_kin_code)), total=kin_data_num):
        if valid[i]:
            continue
        kin_person_indices = np.argwhere(c_personid == kin_id)
        if len(kin_person_indices) == 0:
            kin_error_type[i] = 'Missing'
            continue
        kin_person_indices = kin_person_indices[:, 0]
        kin_person_kin_ids = c_kin_id[kin_person_indices]
        kin_person_kin_codes = c_kin_code[kin_person_indices]
        paired_codes = kincode_pair_dict[kin_code]
        same_kinship = kin_person_kin_ids == personid
        if same_kinship.sum() == 0:
            kin_error_type[i] = 'Missing'
        else:
            valid_pairs = (kin_person_kin_codes.reshape((-1, 1)) == paired_codes.reshape((1, -1))).any(axis=1)
            for j in xrange[kin_person_indices][same_kinship & valid_pairs]:
                valid[j] = True
            for j in xrange[kin_person_indices][same_kinship & ~valid_pairs]:
                kin_error_type[j] = 'Contradictory'
                valid[j] = True
            if (~valid_pairs).all():
                kin_error_type[i] = 'Contradictory'

    kin_data['error_type'] = kin_error_type
    kin_data.to_csv(f"{save_dir}/1_KIN_DATA_NEW_error_type.csv", index=False, encoding='utf-8')

    ################ task 2
    biog_text_data = pd.read_csv("./CBDB_TABLES/BIOG_TEXT_DATA.csv", low_memory=False)
    c_source, c_pages, c_year, c_nh_code, c_nh_year, c_textid, c_role_id = \
        biog_text_data['c_source'], biog_text_data['c_pages'], biog_text_data['c_year'], biog_text_data['c_nh_code'], \
        biog_text_data['c_nh_year'], biog_text_data['c_textid'], biog_text_data['c_role_id']

    def _empty(x):
        return not x or (isinstance(x, float) and np.isnan(x))

    text_error_type = []
    for source, pages, year, nh_code, nh_year, textid, role_id in \
            zip(c_source, c_pages, c_year, c_nh_code, c_nh_year, c_textid, c_role_id):
        error_type = []
        if _empty(source) and not _empty(pages):
            error_type.append('NoSourceHavePage')
        if year > 61:
            error_type.append('HugeNHYear')
        if _empty(nh_code) and not _empty(nh_year):
            error_type.append('NHNoCodeHaveYear')

        text_error_type.append(','.join(error_type))

    text_role_error_type = []
    num = biog_text_data.shape[0]
    i = 0
    while i < num and len(text_role_error_type) < num:
        same_text = [c_role_id[i]]
        for j in range(i+1, num):
            if c_textid[i] != c_textid[j]:
                break
            same_text.append(c_role_id[j])

        cur_num = len(same_text)
        if cur_num == 1:
            # 只有一个人，空
            text_role_error_type.append('')
        elif all(code in [3, 4] for code in same_text) and all(code in same_text for code in [3, 4]):
            # check: 有且仅有3、4两种情况
            text_role_error_type.extend([',Check'] * cur_num)
        elif 0 in same_text:
            # delete: 存在0和其他数字
            text_role_error_type.extend([',Delete'] * cur_num)
        else:
            # unknown: 更复杂的情况，没讨论到
            text_role_error_type.extend([',Unknown'] * cur_num)
        i = j

    biog_text_data['error_type'] = [x + y for x, y in zip(text_error_type, text_role_error_type)]
    biog_text_data.to_csv(f"{save_dir}/2_BIOG_TEXT_DATA_error_type.csv", index=False, encoding='utf-8')

    ################ task 2
    text_codes_data = pd.read_csv("./CBDB_TABLES/TEXT_CODES.csv", low_memory=False)
    c_title_chn, c_title, c_text_nh_year, c_pub_nh_year, c_text_year, c_source, c_pages, c_counter = \
        text_codes_data['c_title_chn'], text_codes_data['c_title'], \
        text_codes_data['c_text_nh_year'], text_codes_data['c_pub_nh_year'], text_codes_data['c_text_year'], \
        text_codes_data['c_source'], text_codes_data['c_pages'], text_codes_data['c_counter']

    def _empty(x):
        return not x or (isinstance(x, float) and np.isnan(x))

    # a-z, A-Z, 半角空格
    en_list = list(range(97, 122+1)) + list(range(65, 90+1)) + [32]
    chn_list = list(range(0x2E80, 0x2FDF+1)) + list(range(0x3400, 0x4DBF+1)) + list(range(0x4E00, 0x9FFF+1))

    text_error_type = []
    for title_chn, title, text_nh_year, pub_nh_year, text_year, source, pages, counter in \
            tqdm(zip(c_title_chn, c_title, c_text_nh_year, c_pub_nh_year, c_text_year, c_source, c_pages, c_counter),
                 total=text_codes_data.shape[0]):
        error_type = []
        if not isinstance(title_chn, str) or any(ord(c) not in chn_list for c in title_chn):
            error_type.append('ErrTitleChn')
        if not isinstance(title, str) or any(ord(c) not in en_list for c in title):
            error_type.append('ErrTitle')
        if text_nh_year > 61:
            error_type.append('HugeTextNHYear')
        if pub_nh_year > 61:
            error_type.append('HugePubNHYear')
        if text_year < 0 or text_year > 2023:
            error_type.append('ErrTextYear')
        if _empty(source) and not _empty(pages):
            error_type.append('NoSourceHavePage')
        if not _empty(counter):
            error_type.append('HaveNum')

        text_error_type.append(','.join(error_type))

    text_codes_data['error_type'] = text_error_type
    text_codes_data.to_csv(f"{save_dir}/3_TEXT_CODES_error_type.csv", index=False, encoding='utf-8')


if __name__ == '__main__':
    # main_7_1()
    # main_7_7()
    # main_7_12()
    # main_7_19()
    # main_7_26()
    # main_8_2()
    main_8_11()



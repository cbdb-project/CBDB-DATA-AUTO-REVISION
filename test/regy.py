import re, os, shutil
import pandas as pd
import copy

def read_directory(directory):
    directory_dict = {}
    for dir_name, subdir_list, file_list in os.walk(directory):
        file_list = [i for i in file_list if not i[0] == '.' and not i[0] == '~']
        if file_list:
            directory_dict[os.path.basename(dir_name)] = file_list
    return directory_dict

def init_output(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for the_file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}: {e}')

def read_regex_patterns(regex_file):
    with open(regex_file, "r", encoding="utf-8") as f:
        regex_string = f.read()
        regex_string = re.sub(r"  +.+", "", regex_string)
        regex_list = regex_string.split("\n")
        for regex_count in list(range(len(regex_list))):
            regex = regex_list[regex_count]
            if regex == "":
                regex_list.remove(regex)
            else:
                regex_list[regex_count] = regex.strip()
                regex_list[regex_count] = re.sub(r"chr(11)", "", regex_list[regex_count])
        return regex_list

def write_output(output, output_dir, output_rest_list, regex_project_name):
    output_df = pd.DataFrame(output, columns=["input_id", "match", "start_location", "input_text", "regex_project_name", "regex_file", "regex_pattern"])
    # output_df.to_excel(f"{output_dir}/{regex_project_name}.xlsx", index=False)
    print(output_df.head())
    # For win
    output_path = os.path.join(output_dir,f"{regex_project_name}.xlsx")
    output_df.to_excel(f"{output_path}.xlsx")
    # output_df.to_excel(f"{output_dir}\{regex_project_name}.xlsx")
    # # For mac
    # output_df.to_excel(f"{output_dir}/{regex_project_name}.xlsx")
    output_rest_list = pd.DataFrame(output_rest_list, columns=["input_id", "input_text_rest"])
    # output_rest_list.to_excel(f"{output_dir}/{regex_project_name}_{OUTPUT_INPUT_REST_NAME}", index=False)
    # For win
    output_rest_path = os.path.join(output_dir,f"{regex_project_name}_{OUTPUT_INPUT_REST_NAME}.xlsx")
    print(output_rest_path)
    print(output_rest_list[:2])
    output_rest_list.to_excel(output_rest_path, engine='xlsxwriter')
    # output_rest_list.to_excel(f"{output_dir}\{regex_project_name}_{OUTPUT_INPUT_REST_NAME}", engine='xlsxwriter')
    # # For mac
    # output_rest_list.to_excel(f"{output_dir}/{regex_project_name}_{OUTPUT_INPUT_REST_NAME}", engine='xlsxwriter')


def data_mining(input_list, regex_dir_dic, output_dir):
    input_total = len(regex_dir_dic)
    inputer_counter = 0
    regex_dir_dic_sorted_keys_list = sorted(regex_dir_dic.keys())
    for regex_project_name in regex_dir_dic_sorted_keys_list:
        output = [] # [input_id, input_text_rest, regex_project_name, regex_file, regex_pattern]
        regex_files = regex_dir_dic[regex_project_name]
        regex_files = sorted(regex_files)
        inputer_counter += 1
        if inputer_counter % 1 == 0:
            print(f"Processing {inputer_counter}/{input_total} subprojects...")
        input_text_rest = ""
        output_rest_list = []
        for input_id, input_text in input_list:
            input_text_rest = copy.deepcopy(input_text)
            for regex_file in regex_files:
                regex_file_path = f"{REGEX_DIR}/{regex_project_name}/{regex_file}"
                regex_list = read_regex_patterns(regex_file_path)
                for regex_pattern in regex_list:
                    matches = re.finditer(regex_pattern, input_text_rest)
                    matches_list = [match.group() for match in matches]
                    matches_for_create_circle = copy.deepcopy(matches_list)
                    matches_for_create_circle = sorted(matches_for_create_circle, key=len, reverse=True)
                    
                    # matches_for_create_circle = re.finditer(regex_pattern, input_text_rest)
                    try:    
                        matches = re.finditer(regex_pattern, input_text_rest)
                    except:
                        print(f"Error in line {input_id}, when using {regex_project_name}/{regex_file}: {regex_pattern}")
                        print("=========================================")
                        break
                    if matches:
                        for match in matches:
                            matched_text = match.group()
                            # whether it have more than one group 分組開關
                            if len(match.groups()) >= 1:
                                matched_text = match.group(1)
                            # whether it have more than one group 分組開關
                            start_location = re.sub("[【】]", "", input_text_rest).find(matched_text)
                            input_text_rest = re.sub(regex_pattern, r"【【\g<0>】】", input_text_rest)
                            output.append([input_id, matched_text, start_location, input_text_rest, regex_project_name, regex_file, regex_pattern])
                            input_text_rest = re.sub("[【】]", "", input_text_rest)
                        # To solove:
                        # 孫二。 and 曾孫二。 can be can be captured by: (?<=[：。？>])(廿九世孫|十二世孫|八世孫|七世孫|五世孫|孫|孫男|男孫|曾孫|曾孫男|曾男孫|男曾孫|重孫男|玄孫|元孫|元孫男|晜孫|裔孫|重孙|嫡孫|長孫|嫡長孫|適孫|次孫|族孫|從孫|聞孫|八孫|女孫|女長孫|孫女|從孫女|曾孫女|女曾孫|外孫|外孫女).+?[。]
                        # If I don't replace key words after finding all the start locations
                        # 孫二 removes 曾孫二 to 曾○○ in the previous loop, then 曾孫二 can't find the start location.
                        for matched_text in matches_for_create_circle:                
                            input_text_rest = input_text_rest.replace(matched_text, len(matched_text)*"○")
            output_rest_list.append([input_id, input_text_rest])
        write_output(output, output_dir, output_rest_list, regex_project_name)

def remove_input_empty_ids_and_texts(input_list):
    for input_id, input_text in list(input_list):
        if input_id != input_id or input_text != input_text:
            input_list.remove([input_id, input_text])
    return input_list

def remove_page_br(input_list):
    output = []
    for input_id, input_text in input_list:
        # input_text = re.sub(r'<page>\d+</page>', '', input_text)
        input_text = re.sub(r'<br ?/>', '', input_text)
        output.append([input_id, input_text])
    return output


REGEX_DIR = "regex"
OUTPUT_DIR = "output"
OUTPUT_INPUT_REST_NAME = "input_rest.xls"
INPUT_FILE = "nanxingqinshu.xlsx"
INPUT_SHEET = "Sheet1"
INPUT_ID_COL = "input_id"
INPUT_TEXT_COL = "match"

init_output(OUTPUT_DIR)
regex_dir_dic = read_directory(REGEX_DIR)
input_list = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)[[INPUT_ID_COL, INPUT_TEXT_COL]].values.tolist()
input_list = remove_input_empty_ids_and_texts(input_list)
input_list = remove_page_br(input_list)
data_mining(input_list, regex_dir_dic, OUTPUT_DIR)
print("Done!")
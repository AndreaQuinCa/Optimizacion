import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from manage_csv import write_csv_from_nested_dict
from manage_csv import read_csv_as_nested_dict


def calculate_statistics(times: np.ndarray, title):
    print(times)
    s = stats.describe(times)._asdict()
    n_dict = {title: s}
    write_csv_from_nested_dict(title, n_dict, s.keys())
    pass


def hyp_test(sample_1, sample_2):
    stats.kruskal(sample_1, sample_2)
    print(stats.kruskal(sample_1, sample_2))
    pass


def update_baseinfo(basedata_dict, base_fieldnames, newinfo_dict, baseinfo):
    """
    Inputs:
      baseinfo_dict - The base data table as dict of dicts dictionary

      newinfo_dict - A dictionary containing the info of the file with new information the keys
                must be contained in the keys of baseinfo

    Output:
          A dictionary
          maps registries to a dictionary containing the items of the two
          dictionaries related to the registries if some key is repeated it stays
          the one from the newinfo[registry] dictionary if the last is not empty
          else, it stays the one in datainfo[registry].
    """
    new_entrys_in_database = set()
    empty_fields = {}
    empty = 'X'
    registry_field = baseinfo['registry']

    for registry, new_registry_dict in newinfo_dict.items():
        if registry in basedata_dict:  # the registry its already in the database
            base_registry_dict = basedata_dict[registry]
            for key, value in new_registry_dict.items():
                if value != '':
                    base_registry_dict[key] = value
                else:
                    if empty_fields.get(registry) == None:
                        empty_fields[registry] = {registry_field: registry}
                    empty_fields[registry][key] = empty
        else:  # a new registry
            new_entrys_in_database.add(registry)
            new_entry = {key: '' for key in base_fieldnames}
            for key, value in new_registry_dict.items():
                new_entry[key] = value
                if value == '':
                    if empty_fields.get(registry) == None:
                        empty_fields[registry] = {registry_field: registry}
                    empty_fields[registry][key] = empty
            basedata_dict[registry] = new_entry

    return basedata_dict, new_entrys_in_database, empty_fields


def write_csv_updated_base(filename_updated, baseinfo, newinfo):
    """
    Inputs:
      filename   - string name of a CSV file that can be rewrited
      baseinfo - A dictionary containing the info of the file of the
                base data of the project it is of the next form (the keys (not the values)
                have to be the same as those of the next example):
                baseinfo = {'filename': 'base_datos.csv', 'registry':'No. Registro'} ### COMPLETAR
      newinfo - A dictionary containing the info of the file with new information the keys
                must be contained in the keys of baseinfo

    Output: It writes two files files: the first its the base info plus the information in the
    the newinfo file it returns an error if a fieldname in the newinfo is not an element of the
    fieldname in the baseinfo
    """
    # Checking if the fieldnames in newinfo file are correct
    base_keyfield = baseinfo['registry']
    new_keyfield = newinfo['registry']

    basedata_dict, base_fieldnames = read_csv_as_nested_dict(baseinfo['filename'],
                                                             base_keyfield)

    newinfo_dict, new_fieldnames = read_csv_as_nested_dict(newinfo['filename'],
                                                           new_keyfield)

    preformat_str_error = "Error: renombra la columna {} del archivo nuevo"
    for key in new_fieldnames:
        if key not in base_fieldnames:
            print(preformat_str_error.format(str(key)))

    # Updating base info
    updated_baseinfo, new_entrys_in_database, empty_fields = update_baseinfo(basedata_dict, base_fieldnames,
                                                                             newinfo_dict, baseinfo)

    # Writing new csv whit the updated info
    # base_fieldnames += ['']
    write_csv_from_nested_dict(filename_updated, updated_baseinfo, base_fieldnames)


def append_dict_to_csv(d_vals, reg, filename):
    d_vals["registry"] = reg
    t_vals = {reg: d_vals}
    write_csv_from_nested_dict(str(reg) + '.csv', t_vals, d_vals.keys())

    newinfo = {'filename': str(reg) + '.csv', 'registry': "registry"}
    baseinfo = {'filename': filename, 'registry': "registry"}

    write_csv_updated_base(filename, baseinfo, newinfo)


def plot_2(x1, y1, x2, y2, data1: str, data2: str, title: str):
    # Add grid
    sns.set_theme()

    plt.plot(x1, y1)
    plt.plot(x2, y2)

    plt.legend([data1, data2])

    # Add the title
    plt.title(title)

    # Save the figure
    plt.savefig(title + '.png')

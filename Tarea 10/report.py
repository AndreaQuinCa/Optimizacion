import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import csv
sns.set_theme()

def read_csv_as_nested_dict(filename, keyfield, separator=',', quote='"'):
    """
    Inputs:
      filename  - Name of CSV file
      keyfield  - Field to use as key for rows
      separator - Character that separates fields
      quote     - Character used to optionally quote fields

    Output: nested dict and a list wiht the fieldnames
      Returns a nested dictionary where the outer dictionary maps the value in the key_field to
      the corresponding row in the CSV file.  The inner dictionaries map the
      field names to the field values for that row.
    """
    with open(filename, "rt", newline='', encoding='utf-8-sig') as csvfile:
        csv_dictdict = {}
        csv_reader = csv.DictReader(csvfile, delimiter=separator, quotechar=quote)
        field_names = csv_reader.fieldnames
        for row in csv_reader:
            row_dict = dict(row)
            csv_dictdict[row_dict.get(keyfield)] = row_dict
    return csv_dictdict, field_names

def read_csv_fieldnames(filename, separator=',', quote='"'):
    """
    Inputs:
      filename  - name of CSV file
      separator - character that separates fields
      quote     - character used to optionally quote fields
    Ouput:
      A list of strings corresponding to the field names in
      the given CSV file.
    """
    with open(filename, "rt", newline='', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=separator, quotechar=quote)
        field_names = csv_reader.fieldnames
    return field_names


def write_csv_from_nested_dict(filename, table, fieldnames, separator=',', quote='"'):
    """
    Inputs:
      filename   - name of CSV file
      table      - list of dictionaries containing the table to write
      fieldnames - list of strings corresponding to the field names in order
      separator  - character that separates fields
      quote      - character used to optionally quote fields
    Output:
      Writes the table to a CSV file with the name filename, using the
      given fieldnames.  The CSV file should use the given separator and
      quote characters.  All non-numeric fields will be quoted.
    """

    with open(filename, "w", newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames, delimiter=separator, quotechar=quote,
                                    quoting=csv.QUOTE_NONNUMERIC)
        head_row = dict(list(zip(fieldnames, fieldnames)))
        csv_writer.writerow(head_row)
        for row in table.values():
            csv_writer.writerow(row)


def calculate_statistics(sample: np.ndarray, title):
    return stats.describe(sample)._asdict()


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
    """
    Añade un renglón a un csv.

    Inputs:
    filename  - name of CSV file
    d_vals    - diccionario con valores que se quieren agregar
                    las llaves deben de ser iguales a los fieldnames del csv
                    al que se añadirá
    reg - el nombre que identificará a la entrada nueva en el csv.
          (debe ser único, si encuentra uno igual en el diccionario,
          lo reescribirá)
    """
    d_vals["registry"] = reg
    t_vals = {reg: d_vals}  # list of dictionaries containing the table to write

    write_csv_from_nested_dict(str(reg) + '.csv', t_vals, d_vals.keys())

    newinfo = {'filename': str(reg) + '.csv', 'registry': "registry"}
    baseinfo = {'filename': filename, 'registry': "registry"}

    write_csv_updated_base(filename, baseinfo, newinfo)


def plot_2(x1, y1, x2, y2, data1: str, data2: str, title: str):

    plt.plot(x1, y1)
    plt.plot(x2, y2)

    plt.legend([data1, data2])

    # Add the title
    plt.title(title)

    # Save the figure
    plt.savefig(title + '.png')

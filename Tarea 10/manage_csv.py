import csv


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


def read_csv_as_nested_dict_test():
    filename = 'test_new_students4.csv'
    keyfield = 'Registro'
    print(read_csv_as_nested_dict(filename, keyfield))


# read_csv_as_nested_dict_test()

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

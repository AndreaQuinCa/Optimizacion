import format_csv as format
import manage_csv as manage
import clean_csv as clean


def update_baseinfo(basedata_dict, base_fieldnames, newinfo_dict, baseinfo):
    """
    Inputs:
      baseinfo_dict - The base data table as dict of dicts dictionary
      
      newinfo_dict - A dictionary containing the info of the file with new information the keys
                must be contained in the keys of baseinfo

    Output:
          A tuple containing two dictionaries and a set.  The first dictionary
          maps registries to a dictionary containing the items of the two
          dictionaries related to the registries if some key is repeated it stays
          the one from the newinfo[registry] dictionary if the last is not empty
          else, it stays the one in datainfo[registry]. The set contains the registries
          from newinfo that were not found in the base data file.
          The second dictionary is a nested one that maps registries to list of keyfields that
          where empty in both data bases (old and new).
      
    """
    new_entrys_in_database = set()
    empty_fields = {}
    empty = 'X'
    registry_field = baseinfo['registry']

    
    for registry, new_registry_dict in newinfo_dict.items():
        if registry in basedata_dict: #the registry its already in the database
            base_registry_dict = basedata_dict[registry]
            for key, value in new_registry_dict.items():
                if value != '':
                    base_registry_dict[key] = value
                else:
                    if empty_fields.get(registry)  == None:
                        empty_fields[registry] = {registry_field:registry}
                    empty_fields[registry][key] = empty
        else: # a new registry
            new_entrys_in_database.add(registry)
            new_entry = {key:'' for key in base_fieldnames}
            for key, value in new_registry_dict.items():
                new_entry[key] = value
                if value == '':
                    if empty_fields.get(registry) == None:
                        empty_fields[registry] = {registry_field:registry}
                    empty_fields[registry][key] = empty
            basedata_dict[registry] = new_entry

    return basedata_dict, new_entrys_in_database, empty_fields


def write_csv_updated_base(filename_updated, filename_emptyfields, baseinfo, newinfo, selected_students):
    
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
    if selected_students:
        basedata_dict, base_fieldnames = clean.clean_base(filename_updated, baseinfo, newinfo)
    else: 
        basedata_dict, base_fieldnames = manage.read_csv_as_nested_dict(baseinfo['filename'],
                                                             base_keyfield)
        
    newinfo_dict, new_fieldnames = manage.read_csv_as_nested_dict(newinfo['filename'],
                                                           new_keyfield)    
    
    preformat_str_error = "Error: renombra la columna {} del archivo nuevo"
    for key in new_fieldnames:
        if key not in base_fieldnames:
            print(preformat_str_error.format(str(key)))   


    # Formating fields
    form_newinfo_dict, unfounded_states = format.table_format(newinfo_dict, newinfo)
    if len(form_newinfo_dict) != len(newinfo_dict):
        print("Error: se perdió información al formatear")
        return None

    # Updating base info
    updated_baseinfo, new_entrys_in_database, empty_fields = update_baseinfo(basedata_dict, base_fieldnames,
                                                                  form_newinfo_dict, baseinfo)
    
    # Writing new csv whit the updated info
    base_fieldnames += ['']
    manage.write_csv_from_nested_dict(filename_updated, updated_baseinfo, base_fieldnames)
    # Writing new csv whit the empty_fields
    new_fieldnames += ['']
    manage.write_csv_from_nested_dict(filename_emptyfields, empty_fields, new_fieldnames)
    
    len_new_entrys = len(new_entrys_in_database)
    len_new_info = len(newinfo_dict)
    if len_new_entrys == len_new_info:
        print('Todas las entradas fueron nuevas')
    elif 0 < len_new_entrys < len_new_info:
        print('Hubieron {} entradas nuevas en la base:\n'.format(len_new_entrys), new_entrys_in_database)
    elif len_new_entrys == 0:
        print('Ninguna entrada fue nueva')
    else:
        print("No sé qué pasó")
    if len(unfounded_states) != 0:    
        print("Clasificar los siguientes estados manualmente:", unfounded_states)

    
def main():
    
    baseinfo = {'filename': 'Infoalumnos.csv', 'registry':'No. Registro',
                'email': 'Dirección de correo electrónico'}
    
    unform_fields = ['Nombre(s)', 'Estado', 'Último grado de estudios', 'Apellido paterno', 'Apellido materno',
                     'Área de trabajo', 'Estado', 'Fecha de nacimiento', 'Sexo', 'Estado',
                     'Último grado de estudios','Título', 'Institución educativa',
                     'Organización(es) para las que trabajas', 'Organización que te postula',
                     'Área de trabajo', 'Municipio']     
    newinfo = {'filename': 'ci-4 seleccionados.csv', 'registry':'No. Registro',
               'unform_fieldnames': unform_fields, 'state': 'Estado', 'email': 'Dirección de correo electrónico'}
  
    str_csv = '.csv'
    filename_updated = baseinfo['filename'][:-4]+'_actualizado'+str_csv
    filename_emptyfields = newinfo['filename'][:-4]+'_faltantes'+str_csv
    write_csv_updated_base(filename_updated, filename_emptyfields, baseinfo, newinfo, True)
    
main()
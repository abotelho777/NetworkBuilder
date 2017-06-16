import numpy as np
import os
import pickle
import psycopg2 as pg
import sys
import warnings
import csv


def getfilenames(directory='./', extension=None):
    names = []
    if extension is None:
        return os.listdir(directory)

    for file in os.listdir(directory):
        if file.endswith(extension):
            names.append(directory + "/" + file)
    return names


def __load_csv__(filename, max_rows=None):
    csvarr = []
    n_lines = len(open(filename).readlines())
    with open(filename, 'r', errors='replace') as f:
        f_lines = csv.reader(f)

        # n_lines = sum(1 for row in f_lines)
        # print(n_lines)
        # n_lines = sum(1 for row in f_lines)
        if max_rows is not None:
            n_lines = max_rows
        sys.stdout.write('-- loading {}...({}%)'.format(filename, 0))
        sys.stdout.flush()
        i = 0
        for line in f_lines:
            # s_line = line.strip()
            # ind = s_line.find('\"')
            # while not ind == -1:
            #     end = s_line.find('\"', ind + 1)
            #     if end == -1:
            #         break
            #     comma = s_line.find(',', ind, end)
            #     while not comma == -1:
            #         s_line = s_line[:comma] + '<comma>' + s_line[comma + 1:]
            #         end = s_line.find('\"', ind + 1)
            #         comma = s_line.find(',', comma, end)
            #     ind = s_line.find('\"', end + 1)
            # # split out each comma-separated value
            # val = s_line.replace('\"','').split(',')
            # for j in range(len(line)):
            #     val[j] = line[j].replace('<comma>', ',')
            #     # try converting to a number, if not, leave it
            #     try:
            #         val[j] = float(val[j])
            #     except ValueError:
            #         # do nothing constructive
            #         pass
            csvarr.append(line)
            if max_rows is not None:
                if len(csvarr) >= max_rows:
                    break
            if not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r-- loading {}...({}%)'.format(filename, round((i / n_lines) * 100, 2)))
                sys.stdout.flush()
            i += 1
        sys.stdout.write('\r-- loading {}...({}%)\n'.format(filename, 100))
        sys.stdout.flush()

    return csvarr


def write_csv(data, filename, headers=None):

    if headers is None:
        headers = []

    with open(filename + '.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator = '\n')

        if len(headers)!=0:
            writer.writerow(np.array(headers, dtype=str))
            # for i in range(0,len(headers)-1):
            #     f.write(str(headers[i]) + ',')
            # f.write(str(headers[len(headers)-1])+'\n')
        # for i in range(0,len(data)):
        ar = np.array(data, dtype=str)
        ar = ar.reshape((ar.shape[0],-1))
        [writer.writerow(j) for j in ar]
        # if len(ar.shape) == 2:
        #     for j in range(0,len(ar[i])-1):
        #         f.write(str(ar[i][j]) + ',')
        #     f.write(str(ar[i][len(ar[i])-1]) + '\n')
        # else:
        #     f.write(str(ar[i]) + '\n')
    f.close()


def read_csv(filename, max_rows=None, headers=True):
    if max_rows is not None:
        max_rows += 1
    data = __load_csv__(filename,max_rows)

    if headers:
        headers = np.array(data[0])
        data = np.delete(data, 0, 0)
        return data, headers
    else:
        return data


def read_csv_headers(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            return line.strip().split(',')
    return []


def pickle_save(instance, filename):
    pickle.dump(instance, open(filename, "wb"), -1)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def infer_if_string(ar, n=None):
    ar = np.array(ar)
    assert len(ar.shape) == 1

    if n is None:
        n = ar.shape[0]
    else:
        n = np.minimum(ar.shape[0],n)

    for i in range(n):
        try:
            float(ar[i])
        except ValueError:
            if ar[i] == '':
                continue
            else:
                return True
    return False


def infer_basic_type(ar, n=None):
    ar = np.array(ar)
    assert len(ar.shape) == 1

    if n is None:
        n = ar.shape[0]
    else:
        n = np.minimum(ar.shape[0],n)

    is_int = True

    for i in range(n):
        try:
            temp = float(ar[i])
            if not temp == int(temp):
                is_int = False
        except ValueError:
            if ar[i] == '':
                continue
            else:
                return 'text'
    return 'double precision' if not is_int else 'int'


def as_factor(ar, return_labels=False):
    ar = np.array(ar).reshape((-1))
    label = np.unique(ar)
    for i in range(len(label)):
        ar[ar[:] == label[i]] = i
    if return_labels:
        lab = [label[int(i)] for i in ar]
        return ar, lab
    return ar


def nan_omit(ar):
    ar = np.array(ar, dtype=str).reshape((-1))
    if not infer_if_string(ar):
        ar = ar[np.where(ar[:] != '')]
        ar = np.array(ar, dtype=np.float32)
        ar = ar[np.where(ar[:] != float('nan'))]
    else:
        ar = ar[np.where(ar[:] != '')]
    return ar

def print_descriptives(ar, headers=None, desc_level=1):
    ar = np.array(ar)
    ar = ar.reshape((-1,ar.shape[-1]))

    if headers is not None:
        assert len(headers) == ar.shape[-1]
        headers = [str(i) + ' ' + headers[i] for i in range(len(headers))]
    else:
        headers = ['Covariate ' + str(i) for i in range(ar.shape[-1])]

    print("{:=<{size}}".format('', size=50 + (30 * desc_level)))
    print("{:<15}{:^25}".format('DESCRIPTIVES', "{} Rows, {} Columns".format(ar.shape[0],ar.shape[1])))
    print("{:=<{size}}".format('', size=50 + (30 * desc_level)))
    for i in range(ar.shape[-1]):
        h = headers[i]
        if len(h) > 15:
            h = ''.join(list(h)[:15]) + '...'
        label = "Column {}".format(i) if headers is None else h
        dtype = ['int','float','string'][np.array(np.where(
            np.array(['int','double precision','text'])[:] == infer_basic_type(ar[:, i], 100))).reshape((-1))[0]]
        label = "{} ({}):".format(label,dtype)

        if dtype == 'string':
            m = np.array(np.array(ar[:, i]) == '').sum()
            desc1 = "{} unique values".format(len(np.unique(ar[:, i])))
            desc2 = ''
            desc3 = ''
        else:
            ar[:, i][ar[:, i] == ''] = float('nan')
            f_ar = np.array(ar[:, i], dtype=np.float32)
            m = np.isnan(f_ar).sum()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                desc1 = "mean={:<.2f} (SD={:<.2f})".format(np.nanmean(f_ar), np.nanstd(f_ar))
                desc2 = "median = {:<.2f}".format(np.nanmedian(f_ar))
                desc3 = "min={:<.2f}, max={:<.2f}".format(np.nanmin(f_ar), np.nanmax(f_ar))
        missing = "{} missing ({:<.1f}%)".format(m, m / float(ar.shape[0]))
        print("{:<30} {:<20} {:<35} {:<30} {:<30}".format(label, missing,
                                                          desc1 if desc_level > 0 else '',
                                                          desc2 if desc_level > 1 else '',
                                                          desc3 if desc_level > 2 else ''))
    print("{:=<{size}}\n".format('', size=50 + (30 * desc_level)))


def db_connect(db_name, user, password='', host='127.0.0.1', port='5432'):
    try:
        return pg.connect(dbname=db_name, user=user, password=password, host=host, port=port)
    except Exception:
        return None


def db_query(db_object, query, arguments=None):
    assert type(arguments) is dict or arguments is None

    cur = db_object.cursor()
    try:
        cur.execute(query, arguments)
    except Exception:
        import traceback
        print('\033[91m')
        traceback.print_exc(file=sys.stdout)
        print(query + '\033[0m')

    try:
        return cur.fetchall()
    except Exception:
        try:
            db_object.commit()
            return []
        except Exception:
            return None


class TableBuilder:
    def __init__(self, name):
        self.fields = []
        self.name = name
        self.num_fields = 0
        self.__primary = False

    def add_field(self, name, type, primary=False):
        if primary:
            assert not self.__primary
            self.__primary = True

        assert type in ['int', 'bigint', 'double precision', 'text', 'timestamp'] # limited data type support
        self.fields.append({'name': name if not primary else 'id',
                            'type': type if not primary else 'bigint',
                            'primary': primary,
                            'values': []})
        self.num_fields += 1

        return self

    def get_fields(self, as_string=False):
        fields = [f['name'] for f in self.fields]
        if not as_string:
            return fields

        f_str = fields[0]
        for f in range(1,len(fields)):
            f_str += ', ' + fields[f]
        return f_str


def db_write_from_csv(filename, db_object, table=None, primary_column=None):
    data, header = read_csv(filename, 100)
    data = np.array(data)
    header = np.array(header)

    primary = primary_column
    if primary_column is None:
        data = np.insert(data,0,range(len(data)),1)
        header = np.insert(header,0,'id')
        primary = 0

    tname = filename[0:-4]
    if table is not None and table.num_fields == 0:
        tname = table.name

    if table is None:
        table = TableBuilder(tname)

    if table.num_fields == 0:
        for i in range(len(header)):
            table.add_field(header[i],infer_basic_type(data.T[i],100),i==primary)

    query = 'DROP TABLE IF EXISTS ' + table.name + ';\n'
    query += 'CREATE TABLE ' + table.name + '('

    for f in range(table.num_fields):
        query += table.fields[f]['name'] + ' ' + table.fields[f]['type'] + ' '
        if table.fields[f]['primary']:
            query += 'PRIMARY KEY'

        if f < table.num_fields - 1:
            query += ', '
        else:
            query += '); '
    print('-- creating table: {}'.format(table.name))
    db_query(db_object, query)

    query = 'INSERT INTO ' + table.name + ' VALUES '

    n_lines = 0

    with open(filename, 'r', errors='ignore') as f:
        f_lines = f.readlines()
        n_lines = len(f_lines)
        sys.stdout.write('-- loading {}...({}%)'.format(filename,0))
        sys.stdout.flush()
        for i in range(1, n_lines):
            line = f_lines[i].strip()

            ind = line.find('\"')
            while not ind == -1:
                end = line.find('\"', ind+1)
                if end == -1:
                    break
                comma = line.find(',',ind, end)
                while not comma == -1:
                    line = line[:comma] + '<comma>' + line[comma + 1:]
                    end = line.find('\"', ind + 1)
                    comma = line.find(',', comma, end)
                ind = line.find('\"', end+1)

            apostrophe = line.find('\'')
            while not apostrophe == -1:
                line = line[:apostrophe] + '<apostrophe>' + line[apostrophe + 1:]
                apostrophe = line.find('\'', apostrophe+len('<apostrophe>'))

            csvalues = np.array(line.replace('\"', '').split(','))

            val = '('

            if primary_column is None:
                val += str(i)
                if len(csvalues) > 0:
                    val += ', '

            for j in range(len(csvalues)):
                csvalues[j] = csvalues[j].replace('<comma>', ',')
                csvalues[j] = csvalues[j].replace('<apostrophe>', '\'\'')
                if table.num_fields == len(csvalues) and table.fields[j]['type'] in ['text', 'timestamp']:
                    val += '\'' + csvalues[j] + '\''
                elif csvalues[j] == '' or csvalues[j] is None:
                    val += 'NULL'
                else:
                    val += csvalues[j]
                if j < len(csvalues) - 1:
                    val += ', '
            val += ')'

            query += val
            if not round((i/n_lines)*100, 2) == round(((i-1)/n_lines)*100, 2):
                sys.stdout.write('\r-- loading {}...({}%)'.format(filename, round((i/n_lines)*100,2)))
                sys.stdout.flush()
                query += ';'
                db_query(db_object, query)
                query = 'INSERT INTO ' + table.name + ' VALUES '
            else:
                query += ', ' if i < n_lines - 1 else ';'
        sys.stdout.write('\r-- loading {}...({}%)\n'.format(filename, 100))
        sys.stdout.flush()

        if query[-1] == ';':
            db_query(db_object, query)
    print('-- {} rows inserted into {}'.format(n_lines, table.name))


def db_create_table(db_object, table):
    assert table.num_fields > 0
    query = 'DROP TABLE IF EXISTS ' + table.name + ';\n'
    query += 'CREATE TABLE ' + table.name + '('

    for f in range(table.num_fields):
        query += table.fields[f]['name'] + ' ' + table.fields[f]['type'] + ' '
        if table.fields[f]['primary']:
            query += 'PRIMARY KEY'

        if f < table.num_fields - 1:
            query += ', '
        else:
            query += '); '
    print('-- creating table: {}'.format(table.name))
    db_query(db_object, query)


def db_write(df, db_object, table, append=False):
    assert type(table) is TableBuilder

    if not append:
        db_create_table(db_object, table)

    df = np.array(df)
    assert len(df.shape) in [1,2]

    if len(df.shape) == 1:
        df = df.reshape((-1,1))

    is_str = [infer_if_string(df[:, i], 100) for i in range(df.shape[1])]

    query = 'INSERT INTO ' + table.name + ' VALUES '
    for i in range(df.shape[0]):
        line = '('
        for j in range(df.shape[1]):
            if (table.num_fields == df.shape[1] and table.fields[j]['type'] in ['text', 'timestamp']) or is_str[j]:
                line += '\'' + df[i,j] + '\''
            elif df[i,j] == '' or df[i,j] is None:
                line += 'NULL'
            else:
                line += df[i,j]
            if j < df.shape[1]-1:
                line += ', '
        line += ')'
        line += ', ' if i < df.shape[0]-1 else ';'
        query += line

    db_query(db_object, query)
    print('-- {} rows inserted into {}'.format(df.shape[0],table.name))

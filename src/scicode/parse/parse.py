from __future__ import annotations

import ast
import json
import re

import h5py
import scipy
import numpy as np
from sympy import Symbol
from datasets import load_dataset

OrderedContent = list[tuple[str, str]]

H5PY_FILE = "eval/data/test_data.h5"

def extract_function_name(function_header):
    pattern = r'\bdef\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    else:
        pattern = r'\bclass\s+(\w+)\s*\('
        match = re.search(pattern, function_header)
        if match:
            return match.group(1)
        else:
            raise ValueError('Function name or class name not found.')

def get_function_from_code(code_string, function_name):
    """
    Extracts and returns the source code of the specified function from a given source code string.

    :param code_string: String containing Python source code
    :param function_name: Name of the function to extract
    :return: String containing the source code of the function, or None if the function is not found
    """
    if code_string is None:
        return None
    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)
        # Iterate through all nodes in the AST
        for node in ast.walk(tree):
            # Check if the node is a function definition
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                # Convert the AST back to a string containing the Python code for the function
                return ast.unparse(node)
    except Exception as e:
        print(f'{function_name} not found with error: {e}')
        return code_string

def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def read_from_hf_dataset(split='validation'):
    dataset = load_dataset('SciCode1/SciCode', split=split)
    return dataset

def rm_comments(string: str) -> str:
    ret_lines = []
    lines = string.split('\n')
    for line in lines:
        if 'matplotlib' in line:
            continue
        if not line.startswith('#'):
            ret_lines.append(line)
    return '\n'.join(ret_lines)


def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num, h5py_file=H5PY_FILE):
    data_lst = []
    with h5py.File(h5py_file, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f'Path {group_path} not found in the file.')
    return data_lst


def save_data_to_hdf5(key, value, h5file):
    if isinstance(value, dict):
        subgroup = h5file.create_group(key)
        save_dict_to_hdf5(value, subgroup)
    elif isinstance(value, (list, tuple)):
        try:
            h5file.create_dataset(key, data=np.array(value))
        except Exception:
            group = h5file.create_group(key)
            subgroup = group.create_group('list')
            for i in range(len(value)):
                save_data_to_hdf5(f'var{i + 1}', value[i], subgroup)
    elif isinstance(value, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix,
                            scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
        group = h5file.create_group(key)
        subgroup = group.create_group('sparse_matrix')
        subgroup.create_dataset('data', data=value.data)
        subgroup.create_dataset('shape', data=value.shape)
        if isinstance(value, scipy.sparse.coo_matrix):
            subgroup.create_dataset('row', data=value.row)
            subgroup.create_dataset('col', data=value.col)
        elif isinstance(value, scipy.sparse.bsr_matrix):
            subgroup.create_dataset('indices', data=value.indices)
            subgroup.create_dataset('indptr', data=value.indptr)
            subgroup.create_dataset('blocksize', data=value.blocksize)
        else:
            subgroup.create_dataset('indices', data=value.indices)
            subgroup.create_dataset('indptr', data=value.indptr)
    elif isinstance(value, (int, float, str, complex, bool,
                            np.bool_, np.int_, np.ndarray)):
        h5file.create_dataset(key, data=value)
    else:
        print(type(value))
        h5file.create_dataset(key, data=str(value))


def save_dict_to_hdf5(data_dict, h5file):
    for key, value in data_dict.items():
        if isinstance(key, (Symbol, np.float_)):
            key = str(key)
        if isinstance(value, dict):
            subgroup = h5file.create_group(key)
            save_dict_to_hdf5(value, subgroup)
        elif isinstance(value, (list, tuple)):
            h5file.create_dataset(key, data=np.array(value))
        elif isinstance(value, (int, float, str, complex, bool,
                                np.bool_, np.ndarray)):
            h5file.create_dataset(key, data=value)
        elif isinstance(value, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix,
                                scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
            maxtrix_group = h5file.create_group(key)
            subgroup = maxtrix_group.create_group('sparse_matrix')
            subgroup.create_dataset('data', data=value.data)
            subgroup.create_dataset('shape', data=value.shape)
            if isinstance(value, scipy.sparse.coo_matrix):
                subgroup.create_dataset('row', data=value.row)
                subgroup.create_dataset('col', data=value.col)
            elif isinstance(value, scipy.sparse.bsr_matrix):
                subgroup.create_dataset('indices', data=value.indices)
                subgroup.create_dataset('indptr', data=value.indptr)
                subgroup.create_dataset('blocksize', data=value.blocksize)
            else:
                subgroup.create_dataset('indices', data=value.indices)
                subgroup.create_dataset('indptr', data=value.indptr)
        else:
            h5file.create_dataset(key, data=str(value))

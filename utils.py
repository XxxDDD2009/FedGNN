import h5py
import numpy as np
import scipy as sp

def load_matlab_file(path_file, name_field):

    #读入数据集，只读模式
    db = h5py.File(path_file, 'r')
    ds = db[name_field]

    #遍历所有的组
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            #print(jc)
            jc = np.asarray(ds['jc'])
            #out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)

    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out
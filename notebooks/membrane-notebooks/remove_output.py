"""
usage: python remove_output.py notebook.ipynb [ > without_output.ipynb ]


"""
import sys
import io
import nbformat
# from nbformat import current
# from IPython.nbformat import current


def remove_outputs(nb):
    """remove the outputs from a notebook"""
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                cell.outputs = []

if __name__ == '__main__':
    fname = sys.argv[1]
    with io.open(fname, 'r') as f:
        nb = nbformat.read(f, 'json')
        # nb = current.read(f, 'json')
    remove_outputs(nb)
    print(nbformat.writes(nb, 'json'))
    # print(current.writes(nb, 'json'))
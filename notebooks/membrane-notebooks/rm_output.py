#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import json
import codecs

def jload(fname):
        return json.load(codecs.open(fname))
def jstore(s,fname,indent=2):
        with codecs.open(fname,"wb",encoding="utf-8") as f:
                json.dump(s,f,indent=indent)

def purge_nb(s):
    def should_keep(o):
        return o.get('output_type') != 'display_data'
    i=0
    # for ws in s['worksheets']:
    # for cell in ws['cells']:
    for cell in s['cells']:
        if cell.get('prompt_number'):
            cell['prompt_number']=i
            i+=1
        os = cell.get('outputs',[])
        os = [o for o in os if should_keep(o)]
        if os:
            cell['outputs'] = os
        else:
            cell.pop('outputs',None)


def main():
    assert len(sys.argv)>=2
    print( sys.argv[1])
    fname =  sys.argv[1]

    s=jload(fname)
    purge_nb(s)
    jstore(s,fname,1)

if __name__ == "__main__":
    sys.exit(main())
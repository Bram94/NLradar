# -*- mode: python -*-

block_cipher = None

import os
import ntpath
import PyQt4
import sys
sys.setrecursionlimit(10000)

a = Analysis(['nlr.py'],
             pathex=['D:\\NLradar\\Python_files',os.path.join(ntpath.dirname(PyQt4.__file__), 'Qt', 'bin')],
             binaries=[],
             datas=[],
             hiddenimports=['pyproj.datadir','h5py.defs','h5py.utils','h5py.h5ac','h5py._proxy','mkl_avx.dll','mkl_def.dll','PyQt5','PyQt5.QtCore','PyQt5.QtGui','PyQt5.QtTest','PyQt5.QtOpenGL'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib','PyQt4'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='nlr',
          debug=False,
          strip=False,
          upx=True,
          console=True,
	  icon='NLradar.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='nlr')

# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""
ptime.py -  Precision time function made os-independent
(should have been taken care of by python)
"""

from __future__ import division

import sys
import time as systime
START_TIME = None
time = None


def winTime():
    """Return the current time in seconds with high precision

    (Windows version, use Manager.time() to stay platform independent.)
    """
    return systime.time()


def unixTime():
    """Return the current time in seconds with high precision

    (Unix version, use Manager.time() to stay platform independent.)
    """
    return systime.time()

if sys.platform.startswith('win'):
    time = winTime
else:
    time = unixTime

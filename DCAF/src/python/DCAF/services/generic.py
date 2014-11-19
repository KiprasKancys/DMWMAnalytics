#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : generic.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Generic service module
"""

# package modules
from DCAF.utils.url_utils import getdata

class GenericService(object):
    "Generic DCAF service class"
    def __init__(self, config=None, verbose=0):
        if  not config:
            config = {}
        self.name = 'generic'
        self.verbose = verbose

    def fetch(self, url, params):
        "Fetch data for given api"
        debug = 0
        if  self.verbose:
            print "GenericService::fetch", url, params
            debug = self.verbose-1
        data = getdata(url, params, debug=debug)
        return data

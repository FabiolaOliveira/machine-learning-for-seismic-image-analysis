#!/usr/bin/env python3

import sys

class t1:
	def __init__(self):
		print('in __init__')
	def __enter__(self):
		print('in __enter__')
	def __exit__(self, extype, exval, traceback):
		print('in __exit__')

with t1() as t:
	print('in with block')
	sys.exit(0)
	print('won\'t execute')


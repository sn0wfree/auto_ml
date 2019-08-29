# -*- coding: utf-8 -*-
# # Copyright by sn0wfree 2018
# ----------------------------
import glob
import importlib
import os
import unittest
import test

# __Version__ = CodersWheel.__Version__
# __Author__ = CodersWheel.__Author__
# __Description__ = CodersWheel.__Description__


def load_modules(path):
	for f in glob.glob(path.rstrip('/')+'/test/*.py'):
		if '__init__.py' not in f:
			name = os.path.splittext(os.path.split(f)[1][0])
			module_name = name
			file_path = f 
			spec = importlib.util.spec_from_file_location(module_name,file_path)
			module = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(module)
			globals()[module_name] = module
			yield module
		

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()
	for m in load_modules(test.__path__[0]):
		suite.addtest(loader.loadTestsFromModule(m))
		
	runner = unittest.TextTestRunner(verbosity=2)
	runner.run(suite)
    

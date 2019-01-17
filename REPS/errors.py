class Error(Exception):
   """Base class for other exceptions"""
   pass

class InvalidPolicyNameError(Error):
   """Somewhere a invalid PolicyName was used"""
   pass
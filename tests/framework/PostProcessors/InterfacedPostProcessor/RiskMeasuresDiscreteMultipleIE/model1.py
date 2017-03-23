def run(self,Input):
  """
    Method that implement a simple system with two components (B and C) in a parallel configuration followed by a component in series (A)
    @ In, Input, dict, dictionary containing the data
    @ Out, outcome, float, logical status of the system given status of the components
  """
  Astatus   = Input['Astatus'][0]
  Bstatus   = Input['Bstatus'][0]
  Cstatus   = Input['Cstatus'][0]

  if (Astatus == 1.0) or (Bstatus == 1.0 and Cstatus == 1.0):
    self.outcome = 1.0
  else:
    self.outcome = 0.0

  return self.outcome
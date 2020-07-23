from pydn.register.models import RegistrationHelper, left_h5_f

rh = RegistrationHelper(left_h5_f, inverse=False)
rh.project_box([444, 527,999,4999])
x=1
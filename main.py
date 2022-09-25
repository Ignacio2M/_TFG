import datetime
import time
import numpy as np

from tranforms.Transform import Transform
from tools.file_tools import ImageTools
from tools.check_point_tools import CheckPoint
time_list = []

IMAGE = 'Patin'


# #####################################################################################################################
start_time = time.time()
file_tool_control = ImageTools(original_dir= f'./Data/Original/LR/{IMAGE}',
                       original_dir_hr= f'./Data/Original/HR/{IMAGE}')

transform_control = Transform(file_tool=file_tool_control, num_samples=10, name=f'{IMAGE}_Test_control_10')

transform_control.No_appy().apply_and_save()
transform_control.proces_net()
transform_control.un_apply(10)
time_list.append(time.time() - start_time)
transform_control.measure()

# #####################################################################################################################
start_time = time.time()
file_tool_control_mini = ImageTools(original_dir= f'./Data/Original/LR/{IMAGE}',
                       original_dir_hr= f'./Data/Original/HR/{IMAGE}')

transform_control_mini = Transform(file_tool=file_tool_control_mini, num_samples=3, name=f'{IMAGE}_Test_control_3')

transform_control_mini.No_appy().apply_and_save()
transform_control_mini.proces_net()
transform_control_mini.un_apply(3)
time_list.append(time.time() - start_time)
transform_control_mini.measure()

start_time = time.time()
file_tool1 = ImageTools(original_dir= f'./Data/Original/LR/{IMAGE}',
                       original_dir_hr= f'./Data/Original/HR/{IMAGE}')

# transform1 = Transform(file_tool=file_tool1, num_samples=10, name=f'{datetime.datetime.today().strftime("%Y%m%d_%H_%M__")}{IMAGE}_Test_Rendimento_SR2_1_3')
transform1 = Transform(file_tool=file_tool1, num_samples=10, name=f'{IMAGE}_Test_RSS_2_1_10')

#
transform1.SR([2,1]).apply_and_save()
transform1.proces_net()
transform1.un_apply(10)
time_list.append(time.time() - start_time)
transform1.measure()

# #####################################################################################################################
start_time = time.time()
file_tool2 = ImageTools(original_dir= f'./Data/Original/LR/{IMAGE}',
                       original_dir_hr= f'./Data/Original/HR/{IMAGE}')

transform2 = Transform(file_tool=file_tool2, num_samples=10, name=f'{IMAGE}_Test_RSS_RS_2_1_10')

transform2.SR([2,1], angle_space_const=False).apply_and_save()
transform2.proces_net()
transform2.un_apply(10)
time_list.append(time.time() - start_time)
transform2.measure()



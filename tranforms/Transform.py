import logging
import os
import time

from TecoGAN.runGan import mycall, mycall_sudo
from tools.file_tools import ImageTools
from tranforms.SR import SR
from tranforms.Vertical_Inter_Split import VIS
from tranforms.No_appy import No_apply
from tranforms.Laplacian import Laplacian
from tranforms.Gaussian_filter import Gaussian
from tools.un_aplly_blocl import UnApplyBlock

from tools.check_point_tools import CheckPoint
import pandas as pd
import matplotlib
from datetime import datetime as dt
from tranforms.Noise import Noise

class Transform:
    num_samples: int
    list_transforms: list
    file_tool: ImageTools

    time_by_block = []
    time_by_proces_net = []

    def __init__(self, file_tool: ImageTools, num_samples: int, name='Test'):
        self.time_by_block=[]
        self.time_by_proces_net=[]
        star_total_time = time.time()
        date = dt.now().strftime("%Y%m%d%H%M%S")
        logging.basicConfig(filename='logs/{DATE}_Transform_{NAME}.log'.format(NAME=name, DATE=date),
                            level=logging.INFO)

        self.num_samples = num_samples
        self.list_transforms = []
        self.file_tool = file_tool
        self.name = name
        # ========== Built samples structure ==========
        self.file_tool.built_structure(num_samples, name)
        self.check_point = CheckPoint(name, file_tool.descript_dit)


    def apply_and_save(self):
        list_results = []
        for trans in self.list_transforms:
            list_results.append(trans.apply())

        self.check_point.save(self)
        logging.info('{NAME} apply ans save'.format(NAME=self.name))

    def proces_net(self, file_load: list or None = None, dir_save: str = None):
        dircst = self.file_tool.list_samples_dir if file_load is None else file_load
        dirstr = self.file_tool.net_dir if dir_save is None else dir_save

        if not os.path.exists(dirstr): os.mkdir(dirstr)

        # run these test cases one by one:
        for index, sample_dir in enumerate(dircst):
            start_time_net_process = time.time()
            print('Procesando: {}/{}'.format(index, len(dircst)))
            name = os.path.splitext(os.path.split(sample_dir)[-1])[0]
            if name != os.path.splitext(os.path.split(dirstr)[-1])[0]:
                out_dir = os.path.join(dirstr, name)
            else:
                out_dir = dirstr
            out_dir_logs = os.path.join(self.file_tool.logs_dir, name)

            if not os.path.exists(out_dir): os.mkdir(out_dir)
            if not os.path.exists(out_dir_logs): os.mkdir(out_dir_logs)

            # Lanza un la ejecucion en un contenedor docker, requiere ser lanzado como root
            cmd1 = ["docker",
                    "exec",
                    "tfg",
                    "python3",
                    "main.py",
                    "--cudaID", "0",  # set the cudaID here to use only one GPU
                    "--output_dir", out_dir,  # Set the place to put the results.
                    "--summary_dir", out_dir_logs + '/',  # Set the place to put the log.
                    "--mode", "inference",
                    "--input_dir_LR", sample_dir,  # the LR directory
                    # "--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
                    # one of (input_dir_HR,input_dir_LR) should be given
                    # "--output_pre", sample_dir,  # the subfolder to save current scene, optional
                    "--num_resblock", "16",  # our model has 16 residual blocks,
                    # the pre-trained FRVSR and TecoGAN mini have 10 residual blocks
                    "--checkpoint", './model/TecoGAN',  # the path of the trained model,
                    "--output_ext", "png"  # png is more accurate, jpg is smaller
                    ]
            with open('./password.txt', 'r') as f:
                sudo_password = f.read()
            mycall_sudo(cmd1).communicate(sudo_password+'\n')

            logging.info(f'Finished nete_process{index}: {time.time() - start_time_net_process}')
            logging.info('{NAME} Proces sample {INDEX}'.format(NAME=self.name, INDEX=index))
            self.time_by_proces_net.append(time.time() - start_time_net_process)

        pd.DataFrame(data=self.time_by_proces_net).to_csv(os.path.join(self.file_tool.measure_dir, 'Time_net.csv'))

    def un_apply(self, block=1):
        if block < 1:
            raise ValueError("El numero de bloques para la desenboltura no puede ser menor que 1")

        list_results = []
        list_reversed_transforms = self.list_transforms
        list_reversed_transforms.reverse()
        # Yamo a la funcion un_apply de cada transformacion.
        for trans in list_reversed_transforms:
            list_results.append(trans.un_apply())

        # consutrullo los bloques de procesado.
        block_samples = []
        num_samples_by_block = int(round(self.num_samples / block))
        num_samples_by_block = list(range(num_samples_by_block, self.num_samples+1, num_samples_by_block))
        if num_samples_by_block[-1] < self.num_samples:
            num_samples_by_block.append(self.num_samples)
        for index, num_samples in enumerate(num_samples_by_block):
            block_samples.append(UnApplyBlock(self.file_tool, num_samples, index))

        self.time_by_block = [0]*len(block_samples)
        for index_dir, dir_name in enumerate(self.file_tool.list_samples_correct_net_dir):
            for index_image, images_path in enumerate(self.file_tool.list_items_from(dir_name)):
                image = self.file_tool.load(images_path)
                for index_block, block_ in enumerate(block_samples):
                    start_time_block = time.time()
                    block_.process(index_dir, index_image, image)
                    finshed_time_bock = time.time() - start_time_block
                    self.time_by_block[block_.id] += finshed_time_bock

        proces_time_dir = []

        for block_ in block_samples:
            block_.save_un_apply()
            proces_time_dir.append({"Bolock_id": block_.id, "Num_images":block_.max_samples*self.file_tool.num_of_images, "Time": self.time_by_block[block_.id]})
            logging.info(f'Finished un_apply_bloc{block_.id}_{block_.max_samples}: {self.time_by_block[block_.id]}')

        pd.DataFrame(data=proces_time_dir).to_csv(os.path.join(self.file_tool.measure_dir, 'Time_un_apply.csv'))

    def measure(self):
        if not self.file_tool.exists_control_metrics:
            if not self.file_tool.exists_control_dir:
                self.proces_net([self.file_tool.original_dir], self.file_tool.control_net_dir)
            self._measures([self.file_tool.control_net_dir], self.file_tool.control_metric_dir,
                           [self.file_tool.original_dir_hr])

        sub_samples = os.listdir(self.file_tool.final_dir)


        for sample in sub_samples:
            out_dir = self.file_tool.create_dir(os.path.join(self.file_tool.measure_dir, sample))
            sample = os.path.join(self.file_tool.final_dir, sample)
            self._measures([sample], out_dir, [self.file_tool.original_dir_hr])
            # self._graph(self.file_tool.control_metric_dir, out_dir)

        sample = list(map(lambda x: os.path.join(self.file_tool.final_dir, x), sub_samples))
        out_dir = self.file_tool.create_dir(self.file_tool.measure_dir)
        # sample = os.path.join(self.file_tool.final_dir, sample)
        self._measures(sample, out_dir, [self.file_tool.original_dir_hr]*len(sample))
        # self._graph(self.file_tool.control_metric_dir, out_dir)



    def _measures(self, testpre, dirstr, tarstr):
        cmd1 = ["python3", "TecoGAN/metrics.py",
                "--output", dirstr,
                "--results", ",".join(testpre),
                "--targets", ",".join(tarstr),
                ]
        mycall(cmd1, True).communicate()
        logging.info('{NAME} Generate measure testpre:{testpre}, dirstr:{dirstr}, tarstr:{tarstr}'.format(NAME=self.name,testpre=testpre, dirstr=dirstr, tarstr=tarstr))

    # def _graph(self, control_dir, measure_dir):
    #     # self.file_tool.control_metric_dir
    #     df = pd.read_csv(control_dir + '/metrics.csv')
    #     # self.file_tool.measure_dir
    #     df2 = pd.read_csv(measure_dir + '/metrics_0.csv')
    # 
    #     df.drop(columns='Unnamed: 0', inplace=True)
    #     df2.drop(columns='Unnamed: 0', inplace=True)
    # 
    #     df_copy = df.drop(index=list(range(len(df.index) - 6, len(df.index), 1))).astype('float')
    #     df_copy2 = df2
    # 
    #     dfd = df_copy.describe().loc[['mean', 'std']]
    #     dfd2 = df_copy2.describe().loc[['mean', 'std']]
    # 
    #     PSNR_df = pd.DataFrame([dfd["PSNR_00"], dfd2["PSNR_00"]]).T
    #     PSNR_df.columns = ['Original', 'Apply_tranform']
    #     SSIM_df = pd.DataFrame([dfd["SSIM_00"], dfd2["SSIM_00"]]).T
    #     SSIM_df.columns = ['Original', 'Apply_tranform']
    #     LPIPS_df = pd.DataFrame([dfd["LPIPS_00"], dfd2["LPIPS_00"]]).T
    #     LPIPS_df.columns = ['Original', 'Apply_tranform']
    #     tOF_df = pd.DataFrame([dfd["tOF_00"], dfd2["tOF_00"]]).T
    #     tOF_df.columns = ['Original', 'Apply_tranform']
    #     tLP100_df = pd.DataFrame([dfd["tLP100_00"], dfd2["tLP100_00"]]).T
    #     tLP100_df.columns = ['Original', 'Apply_tranform']
    #     logging.info('{NAME} generate graph'.format(NAME=self.name))
    # 
    #     # def grafi(plots):
    #     #     for bar in plots.patches:
    #     #         # Using Matplotlib's annotate function and
    #     #         # passing the coordinates where the annotation shall be done
    #     #         plots.annotate(format(bar.get_height(), '.2f'),
    #     #                        (bar.get_x() + bar.get_width() / 2,
    #     #                         bar.get_height()), ha='center', va='center',
    #     #                        size=15, xytext=(0, 5),
    #     #                        textcoords='offset points')
    #     #     return plots
    # 
    #     # matplotlib.style.use('fivethirtyeight')
    #     # PSNR_df.plot.bar(title='PSNR')
    #     # 
    #     # grafi(PSNR_df.plot.bar(title='PSNR')).get_figure().savefig(measure_dir + '/PSNR_df.pdf', dpi=300,
    #     #                                                            bbox_inches='tight')
    #     # grafi(SSIM_df.plot.bar(title='SSIM')).get_figure().savefig(measure_dir + '/SSIM_df.pdf', dpi=300,
    #     #                                                            bbox_inches='tight')
    #     # grafi(LPIPS_df.plot.bar(title='LPIPS')).get_figure().savefig(measure_dir + '/LPIPS_df.pdf',
    #     #                                                              dpi=300,
    #     #                                                              bbox_inches='tight')
    #     # grafi(tOF_df.plot.bar(title='tOF')).get_figure().savefig(measure_dir + '/tOF_df.pdf', dpi=300,
    #     #                                                          bbox_inches='tight')
    #     # grafi(tLP100_df.plot.bar(title='tLP100')).get_figure().savefig(measure_dir + '/tLP100_df.pdf',
    #     #                                                                dpi=300,
    #     #                                                                bbox_inches='tight')

    def No_appy(self):
        pre = No_apply()
        self.list_transforms.append(pre)
        logging.info('{NAME} add No_apply layer'.format(NAME=self.name))
        return self

    def SR(self, shift_vector, angle_time_const=True, shift_time_const=True, angle_space_const=True,
           shift_space_const=False, black_backraun=False):
        pre = SR(shift_vector=shift_vector,
                 file_tool=self.file_tool,
                 angle_time_const=angle_time_const,
                 shift_time_const=shift_time_const,
                 angle_space_const=angle_space_const,
                 shift_space_const=shift_space_const,
                 black_backraun = black_backraun)

        self.list_transforms.append(pre)
        logging.info('{NAME} add SR layer'.format(NAME=self.name))
        return self

    def VIS(self, split_num, increment_step):
        pre = VIS(file_tool=self.file_tool, split_num=split_num, increment_step=increment_step)
        self.list_transforms.append(pre)
        logging.info('{NAME} add VIS layer'.format(NAME=self.name))
        return self

    def Gaussian(self, kernel, sigma_x, random_apply=None):
        pre = Gaussian(file_tool=self.file_tool, kernel=kernel, sigma_x=sigma_x, random_apply=random_apply)
        self.list_transforms.append(pre)
        logging.info('{NAME} add Gaussian layer'.format(NAME=self.name))
        return self

    def Noise(self, noise_type, random_apply=None):
        pre = Noise(file_tool=self.file_tool, noise_type= noise_type, random_apply= random_apply)
        self.list_transforms.append(pre)
        logging.info('{NAME} add Noise layer'.format(NAME=self.name))
        return self

    def Laplacian(self, random_apply=None):
        pre = Laplacian(file_tool=self.file_tool, random_apply= random_apply)
        self.list_transforms.append(pre)
        logging.info('{NAME} add Laplacian layer'.format(NAME=self.name))
        return self

    

    def info(self):
        return {
            'name': self.name,
            'num_samples': self.num_samples,
            'list_transforms': [item.info() for item in self.list_transforms],
            'files': self.file_tool.info()
        }

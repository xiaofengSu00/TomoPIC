import os 
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

### read particle coordinate
def read_data(data_dir,out_file):
    particle_dict = {'4CR2': [1, 22],
			'1QVR': [2, 17],
			'1BXN': [3, 12],
			'3CF3': [4, 14],
			'1U6G': [5, 12],
			'3D2F': [6, 13],
			'2CG9': [7, 12],
			'3H84': [8, 14],
			'3GL1': [9, 7],
			'3QM1': [10, 7],
			'1S3X': [11, 6],
			'5MRC': [12, 28],
			'fiducial': [13, 12]}
    # particle_dict = {'3cf3': [1, 14],
    #                           '1s3x': [2, 6],
    #                           '1u6g': [3, 13],
    #                           '4cr2': [4, 24],
    #                           '1qvr': [5, 17],
    #                           '3h84': [6, 14],
    #                           '2cg9': [7, 12],
    #                           '3qm1': [8, 7],
    #                           '3gl1': [9, 7],
    #                           '3d2f': [10, 13],
    #                           '4d8q': [11, 16],
    #                           '1bxn': [12, 12]}
    class_list = []
    radius_list = []

    for i in range(10):
       location = pd.read_csv(os.path.join(data_dir, 'model_%d/particle_locations.txt' % i), header=None, sep=' ')
       for j in range(len(location)):
            particle = location.loc[j]
            if particle[0] not in particle_dict.keys():
                continue
            p_class, p_size = particle_dict[particle[0]]

            class_list.append(p_class)
            radius_list.append(p_size)
    
    data = {'class':class_list,
            'radius':radius_list}
    df = pd.DataFrame(data)
    df.to_csv(out_file,index=False,sep=' ')

def cluster(file,n):
    data = pd.read_csv(file, header=0,sep = ' ')
    p_size = data['radius'].to_numpy().reshape(-1,1)
    # 设置K-means聚类的数量
    
    # 创建KMeans对象并拟合数据
   # 选择簇的数量
    kmeans = KMeans(n_clusters=n)
    # 拟合模型
    kmeans.fit(p_size)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 输出聚类结果
    print("Labels:", labels)
    print("Cluster Centers:", centers)

if __name__ == '__main__':
    data_dir = '/storage_data/su_xiaofeng/shrec_data/shrec_2021'
    out_file = '/storage_data/su_xiaofeng/shrec_data/shrec_2021/all_patilces_class13_mf.csv'

    # data_dir = '/storage_data/su_xiaofeng/shrec_data/shrec_2020/shrec2020_full_dataset'
    # out_file = '/storage_data/su_xiaofeng/shrec_data/shrec_2020/shrec2020_full_dataset/all_patilces_class13_mf.csv'

    read_data(data_dir,out_file)

    cluster(out_file,n=3)
    print('-------')

    ### 3 [34.5,20.3,13.5]  Cluster Centers: [[34.54938272] [20.33765191] [13.47494442]]
    ### 4 Cluster Centers: [[17.49802711][34.54938272][12.63229927][22.68067034]] 
    ### 5 Cluster Centers: [[17.49802711] [34.54938272] [12.63229927] [25.        ] [21.4944483 ]]
    #### [13,20,35] [13,17,23,35], [13,17,21,25,35]  7,13,25//// 6,13,19,28
    ### 6,11,16--- 7,14,24
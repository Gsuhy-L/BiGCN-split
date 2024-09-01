import numpy as np
import os
from collections import defaultdict


def get_twitter_split(dataset_name,count_min=0,count_max=101,interv = 5):
    Project_path = "/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/"

    # rvnn_path = Project_path + "rumor_detection_acl2017/"
    bigcn_path = Project_path + "Process/data/"
    bigcn_split_path = Project_path + "Process/count_split/"

    # print("Loading {}".format(path))
    # path_rvnn = rvnn_path+dataset_name+"matrix/"
    path_bigcn = bigcn_path + dataset_name + "fulltime/"

    if path_bigcn[-1] == '/':
        # 把

        bigcn_files = sorted([path_bigcn + f for f in os.listdir(path_bigcn)],
                             key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
        # print(bigcn_files)
        # if "Pheme" in dataset_name:
        #     bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/498300832648273920.npz")
        #     bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/525068253341970432.npz")
        #     bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/553540824768991233.npz")
        #     bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/580352540316946432.npz")
        # elif "Twitter" in dataset_name:

        count = 0

        #依次遍历每个事件
        for i in range(len(bigcn_files)):
            # print('++++++++++++++++++++++++')
            # try:
            # if
            time_count = {}
            # rvnn_file = bigcn_files[i]
            #取每个事件
            bigcn_file = bigcn_files[i]
            # if  "547509900130021377"  not in bigcn_file:
            #     continue
            #取事件名字
            id = int(bigcn_file.split('/')[-1].split('.')[0])
            print(bigcn_file)

            # rvnn_file_data = np.load(os.path.join(rvnn_file), allow_pickle=True)
            bigcn_file_data = np.load(os.path.join(bigcn_file), allow_pickle=True)

            # TODO rvnn里边还包括了时间数组
            # rvnn_num, rvnn_matrix, rvnn_root = rvnn_file_data['num'], rvnn_file_data['edgeindex'], rvnn_file_data[
            #     'rootindex']
            #TODO 取出每个事件对应的内容
            bigcn_matrix, bigcn_root, bigcn_time, bigcn_x, bigcn_y = bigcn_file_data['edgeindex'], \
                                                                     bigcn_file_data['rootindex'], bigcn_file_data[
                                                                         'edgetime'], bigcn_file_data['x'], \
                                                                     bigcn_file_data['y']
            print(bigcn_matrix)
            print(bigcn_root)
            print(bigcn_time)

            if len(bigcn_time) == 0:
                continue

            # 0-36
            for i in range(count_min,count_max,interv):
                zero_save_dir = os.path.join(bigcn_split_path, dataset_name + '/' + str(i) + '/')
                if not os.path.exists(zero_save_dir):
                    os.makedirs(zero_save_dir)
                # 这里有一点问题
                np.savez(os.path.join(zero_save_dir + str(id) + '.txt'),
                         edgeindex=np.array([[], []]), rootindex=0, x=np.array([bigcn_x[bigcn_root]]), y=bigcn_y)
            time_flag = 0
            for str_time in bigcn_time:
                #只输出根节点的时间
                #print(str_time)
                #TODO 如果存在节点的时间在我们要求的时间之前，我们可以继续操作
                if float(str_time)<=float(10*5):
                    time_flag=1
                    break
                    #TODO 否则，不对该文件chuli
            if time_flag == 0:
                continue


            max_idx = 0
            max_time = 0
            count_split_set = defaultdict(list)
            #TODO 找到时间节点集合

            # for post_time_index in range(len(bigcn_time)):
            #     if float(bigcn_time[post_time_index]) > max_time:
            #         max_idx = post_time_index
                    # max_time = float(bigcn_time[max_idx])
            # split_num = int(max_time // 5)
            # res_num = max_time % 5

            # max_split = 10

            # if len
            for count_num in range(count_min, count_max, interv):
                time_idx_map = {}
                # post_time = float(bigcn_time[post_time_index])

                #TODO
                time_sort_lst = []
                idx_sort_lst = []
                #for post_time_index in range(len(bigcn_time)):
                for node_index, node_time in enumerate(bigcn_time):
                    time_idx_map[node_index] = float(node_time)
                #TODO
                time_sorted_items = sorted(time_idx_map.items(),key = lambda x:x[1])
                print(time_sorted_items)
                # print(e)
                #for item in sorted_items
                if len(time_sorted_items)<=count_num:
                    for post_time_index in range(len(bigcn_time)):
                        if post_time_index == 0:
                            continue
                        count_split_set[count_num].append([bigcn_matrix[0][post_time_index-1], bigcn_matrix[1][post_time_index-1]])
                else:
                    for item in time_sorted_items[:count_num]:
                        # print(time_sorted_items)
                        # print(bigcn_matrix)
                        # print(item[0])
                        if item[0] == bigcn_root:
                            continue

                        count_split_set[count_num].append([bigcn_matrix[0][item[0]-1], bigcn_matrix[1][item[0]-1]])
                # print()
                # print(count_split_set)
                # print(e)
                #select_idx = time_sorted_items[]

                    # 把数据放到对应时间片的字典数组中
                    # print(i)
                    #临时可
                    # if post_hour < i:
                    #     time_split_set[i].append([bigcn_matrix[0][post_time_index], bigcn_matrix[1][post_time_index]])
            # print(count_split_set)
            for key, val in count_split_set.items():
                print(val)
                # print()
                # print('key',key)
                # print(val)
                split_matrix_list_row = []
                split_matrix_list_col = []
                # if len()
                # print(val)
                for matrix in val:
                    split_matrix_list_row.append(matrix[0])
                    split_matrix_list_col.append(matrix[1])
                #TODO 保存的是对应不同的时间的节点连边
                edgeindex = [split_matrix_list_row, split_matrix_list_col]
                # if len()
                #time_count[key] = 1
                save_dir = os.path.join(bigcn_split_path, dataset_name + '/' + str(key) + '/')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # print(edgeindex)
                # max_node = max(max(edgeindex[0]), max(edgeindex[1]))
                node_list = {}
                new_edge_row = []
                new_edge_col = []
                new_bigcn_x = []
                # print(len(edgeindex[0]))
                # print(len(bigcn_time))
                for edge_row in edgeindex[0]:
                    #如果头节点没有被添加
                    if edge_row not in node_list:
                        # print(node_list)
                        #先添加头节点
                        node_list[edge_row] = len(node_list)
                        new_bigcn_x.append(bigcn_x[edge_row])
                    #再把对应的特征添加进去
                    new_edge_row.append(node_list[edge_row])

                for edge_col  in edgeindex[1]:
                    #TODO 同理，如果列节点的值没有被添加，就把列的值也添加进去
                    if edge_col not in node_list:
                        node_list[edge_col]=len(node_list)
                        new_bigcn_x.append(bigcn_x[edge_col])
                    new_edge_col.append(node_list[edge_col])
                print('=====================')
                print(node_list)
                new_edgeindex = [new_edge_row,new_edge_col]
                new_edgeindex = np.array(new_edgeindex)
                new_bigcn_x = np.array(new_bigcn_x)
                #print(len(new_bigcn_x))
                # print(node_list)


                new_bigcn_root = np.array(node_list[int(bigcn_root)])
                # print(new_edgeindex)
                # print(e)
                # print(e)

    #TODO 先把生成文件的部分给注释了
                np.savez(os.path.join(save_dir + str(id) + '.txt'),
                         edgeindex=new_edgeindex, rootindex=new_bigcn_root, x=new_bigcn_x, y=bigcn_y)

        print(count)


get_twitter_split(dataset_name='Pheme')


import numpy as np
import os
from collections import defaultdict


def get_twitter_split(dataset_name):
    Project_path = "/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/"

    # rvnn_path = Project_path + "rumor_detection_acl2017/"
    bigcn_path = Project_path + "Process/data/"
    bigcn_split_path = Project_path + "Process/data_split/"

    # print("Loading {}".format(path))
    # path_rvnn = rvnn_path+dataset_name+"matrix/"
    path_bigcn = bigcn_path + dataset_name + "fulltime/"

    if path_bigcn[-1] == '/':
        # 把

        bigcn_files = sorted([path_bigcn + f for f in os.listdir(path_bigcn)],
                             key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
        count = 0
        for i in range(len(bigcn_files)):
            # print('++++++++++++++++++++++++')
            # try:
            # if
            time_count = {}
            # rvnn_file = bigcn_files[i]
            bigcn_file = bigcn_files[i]
            # if  "547509900130021377"  not in bigcn_file:
            #     continue
            id = int(bigcn_file.split('/')[-1].split('.')[0])
            print(bigcn_file)

            # rvnn_file_data = np.load(os.path.join(rvnn_file), allow_pickle=True)
            bigcn_file_data = np.load(os.path.join(bigcn_file), allow_pickle=True)
            # np.savez(os.path.join(cwd, 'data/' + obj + 'matrix/' + id + '.txt'), num=rootfeat.shape[0], edgeindex=tree,
            #          rootindex=rootindex)
            # np.savez(os.path.join(rvnn_path,  dataset_name + 'matrix/' + str(id) + '.txt'), num=idx, edgeindex=tree,
            #          rootindex=rootindex)
            # TODO rvnn里边还包括了时间数组
            # rvnn_num, rvnn_matrix, rvnn_root = rvnn_file_data['num'], rvnn_file_data['edgeindex'], rvnn_file_data[
            #     'rootindex']
            bigcn_matrix, bigcn_root, bigcn_time, bigcn_x, bigcn_y = bigcn_file_data['edgeindex'], \
                                                                     bigcn_file_data['rootindex'], bigcn_file_data[
                                                                         'edgetime'], bigcn_file_data['x'], \
                                                                     bigcn_file_data['y']
            #
            # print(bigcn_time)
            # 还要考虑内容为空的情况

            if len(bigcn_time) == 0:
                continue

            # 0-36
            for i in range(37):
                zero_save_dir = os.path.join(bigcn_split_path, dataset_name + '/' + str(i) + '/')
                if not os.path.exists(zero_save_dir):
                    os.makedirs(zero_save_dir)
                # 这里有一点问题
                np.savez(os.path.join(zero_save_dir + str(id) + '.txt'),
                         edgeindex=np.array([[], []]), rootindex=0, x=np.array([bigcn_x[bigcn_root]]), y=bigcn_y)
            time_flag = 0
            for str_time in bigcn_time:

                if float(str_time)<=float(37*5):
                    time_flag=1
                    break
            if time_flag == 0:
                continue


            max_idx = 0
            max_time = 0
            time_split_set = defaultdict(list)
            for post_time_index in range(len(bigcn_time)):
                if float(bigcn_time[post_time_index]) > max_time:
                    max_idx = post_time_index
                    max_time = float(bigcn_time[max_idx])
            split_num = int(max_time // 5)
            res_num = max_time % 5

            max_split = 37
            # print(split_num)
            # if split_num >36:
            #     real_split_num = 36
            # else:
            #      real_split_num = split_num
            # 36个小时，那个就有37个节点
            # if len
            for post_time_index in range(len(bigcn_time)):
                post_time = float(bigcn_time[post_time_index])
                post_hour = int(post_time // 5)
                post_rest = post_time%5.0
                if post_rest != 0.:
                    post_hour+=1
                # print(post_hour)
                # print(time_split_set[0])
                # 0-36

                for i in range(max_split):
                    # 把数据放到对应时间片的字典数组中
                    # print(i)
                    #临时可
                    if post_hour < i:
                        time_split_set[i].append([bigcn_matrix[0][post_time_index], bigcn_matrix[1][post_time_index]])
                # if post_hour >= max_split:
                #     time_split_set[max_split].append(
                #         [bigcn_matrix[0][post_time_index], bigcn_matrix[1][post_time_index]])

            # time_split_matrix_set = defaultdict(list)
            # if len()
            # if len(time_split_set)<36:
            #     print(bigcn_file)
            #     break

            # if len(time_split_set) == 0:
            #
            #     print(bigcn_file)

            # count+=1

            # if len(time_split_set[0]) == 0:
            # print(bigcn_file)
            # print(time_split_set[0])
            # print(len(bigcn_matrix[0]))
            # print(len(bigcn_time))
            # print(e)
            # print(len(bigcn_matrix[0]))
            # print(len(bigcn_matrix[0]))
            # print(len(bigcn_time))
            # print(bigcn_matrix)
            # print(bigcn_time)
            for key, val in time_split_set.items():
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
                # print('-----------------------')
                # print(key)
                # print(split_matrix_list_row)
                # print(split_matrix_list_col)
                # new_bigcn_time = np.array(new_bigcn_time)
                edgeindex = [split_matrix_list_row, split_matrix_list_col]
                # if len()
                time_count[key] = 1
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
                    if edge_row not in node_list:
                        # print(node_list)
                        node_list[edge_row] = len(node_list)
                        new_bigcn_x.append(bigcn_x[edge_row])
                    new_edge_row.append(node_list[edge_row])

                for edge_col  in edgeindex[1]:
                    if edge_col not in node_list:
                        node_list[edge_col]=len(node_list)
                        new_bigcn_x.append(bigcn_x[edge_col])
                    new_edge_col.append(node_list[edge_col])
                # print('=====================')
                # print(key)
                # print(bigcn_root)
                # print(bigcn_matrix)
                # print(bigcn_time)
                # print(edgeindex[0])
                # print(edgeindex[1])
                # print(new_edge_row)
                # print(new_edge_col)
                # print()
                # print(node_list)
                new_edgeindex = [new_edge_row,new_edge_col]
                new_edgeindex = np.array(new_edgeindex)
                new_bigcn_x = np.array(new_bigcn_x)
                # print(node_list)

                new_bigcn_root = np.array(node_list[int(bigcn_root)])

                np.savez(os.path.join(save_dir + str(id) + '.txt'),
                         edgeindex=new_edgeindex, rootindex=new_bigcn_root, x=new_bigcn_x, y=bigcn_y)
            # print('ok')
            # if len(time_count)<37:
            #     print(bigcn_file)
            #     count+=1
            # print(time_split_set)
            # print(len(time_count))
            # print(e)
        print(count)


get_twitter_split(dataset_name='Pheme')


import numpy as np
import os
from collections import defaultdict


def get_twitter_split(dataset_name,num_min,num_max):
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
        # if "Pheme" in dataset_name:
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/498300832648273920.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/525068253341970432.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/553540824768991233.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/580352540316946432.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/499362441697193985.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/499365436816105473.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/500375737439096833.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/524931830173421568.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/524935023380545537.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/524958383468978176.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/524970488590651395.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/544288302254137345.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/544323450270392322.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/544346359948910592.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/544436750097973250.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/544480083982168064.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/544513339544838147.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/544518445870497793.npz")
            # bigcn_files.remove("/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/data/Phemefulltime/553145034720030720.npz")
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
            print(bigcn_file_data)
            bigcn_matrix, bigcn_root, bigcn_time, bigcn_x, bigcn_y = bigcn_file_data['edgeindex'], \
                                                                     bigcn_file_data['rootindex'], bigcn_file_data['edgetime'], bigcn_file_data['x'], bigcn_file_data['y']
            new_bigcn_time = []
            print(bigcn_time)
            # if "499706791396392960" in bigcn_file:
            for time in bigcn_time:
                post_min=int(float(time)//60)
                min_rest = float(time)%60.0
                # print(time)
                # print(post_min)
                # print(min_rest)

                if min_rest != 0:
                    post_min+=1
                # print(post_min)
                # print('===========')

                new_bigcn_time.append(post_min)
            bigcn_time = new_bigcn_time

            #
            print(bigcn_time)
                # print(e)
            # 还要考虑内容为空的情况

            if len(bigcn_time) == 0:
                continue

            # 0-36
            for i in range(num_min,num_max):
                zero_save_dir = os.path.join(bigcn_split_path, dataset_name + '/' + str(i) + '/')
                if not os.path.exists(zero_save_dir):
                    os.makedirs(zero_save_dir)
                # 这里有一点问题
                np.savez(os.path.join(zero_save_dir + str(id) + '.txt'),
                         edgeindex=np.array([[], []]), rootindex=0, x=np.array([bigcn_x[bigcn_root]]), y=bigcn_y)
            time_flag = 0
            for str_time in bigcn_time:

                if float(str_time)<=float(num_max*60):
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
            # split_num = int(max_time // 5)
            # res_num = max_time % 5
            #
            max_split = num_max
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
                if post_rest != 0:
                    post_hour+=1
                # print(post_hour)
                # print(time_split_set[0])
                # 0-36

                for i in range(num_min,max_split):
                    # 把数据放到对应时间片的字典数组中
                    # print(i)
                    #临时可
                    if post_hour < i:
                        # if "553490238979727360" in bigcn_file:
                        #     print(bigcn_matrix)
                        #     print(bigcn_time)
                        #     print(post_time_index)
                        #     print(post_hour)
                        #     print(i)
                        #     print(e)
                        if post_time_index == 0:
                            continue

                        time_split_set[i].append([bigcn_matrix[0][post_time_index-1], bigcn_matrix[1][post_time_index-1]])
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
                # print(edgeindex)a
                # max_node = max(max(edgeindex[0]), max(edgeindex[1]))
                node_list = {}
                new_edge_row = []
                new_edge_col = []
                new_bigcn_x = []

                # print(len(edgeindex[0]))
                # print(len(bigcn_time))
                # if "499448210789007360" in bigcn_file:
                #     print(edgeindex)
                #     print(e)
                node_lst = []
                # for i in edgeindex[0]:
                #     if i not in node_lst:
                #         node_lst.append(i)
                # for j in edgeindex[1]:
                #     if j not in node_lst:
                #         node_lst.append(j)
                for i in range(len(edgeindex[0])):
                    if edgeindex[0][i] not in node_lst:
                        node_lst.append(edgeindex[0][i])

                    if edgeindex[1][i] not in node_lst:
                        node_lst.append(edgeindex[1][i])
                # print(node_lst)
                node_redict = {id:i for i,id in enumerate(node_lst)}
                new_bigcn_x = bigcn_x[node_lst,:]
                new_edge_row = [node_redict[tup] for tup in edgeindex[0]]
                new_edge_col = [node_redict[tup] for tup in edgeindex[1]]

                # for edge_row in edgeindex[0]:
                #     if edge_row not in node_list:
                #         # print(node_list)
                #         node_list[edge_row] = len(node_list)
                #         new_bigcn_x.append(bigcn_x[edge_row])
                #     new_edge_row.append(node_list[edge_row])
                #
                # for edge_col  in edgeindex[1]:
                #     if edge_col not in node_list:
                #         node_list[edge_col]=len(node_list)
                #         new_bigcn_x.append(bigcn_x[edge_col])
                #     new_edge_col.append(node_list[edge_col])
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

                new_bigcn_root = np.array(node_redict[int(bigcn_root)])

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


get_twitter_split(dataset_name='Pheme',num_min=0,num_max=20)


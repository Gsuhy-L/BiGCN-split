
import os
import tqdm
root_path = "/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/"
file_path = root_path+'rumor_detection_acl2017/Twitter15/tree/'


def split_post_by_time(path):
    print("Loading {}".format(path))
    if path[-1] == '/':
        files = sorted([path + f for f in os.listdir(path)],
                       key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
        # files = files[: DEBUG_NUM] if self.params.DEBUG else files
        # data = [read_and_unpkl(file) for file in tqdm(files)]
    # else:
    #     with open(path, 'rb') as f:
    #         data = pkl.load(f)
    for file in files:
        f = open(file,'r').readlines()
        #文件名对应一个时间
        file_name = file.split('/')[-1].split('.')[0]
        for line in f:
            #取一个事件中的所有帖子
            split_line = eval(line.split('->')[1])
            # print(split_line)
            # print(split_line[2])
            # print(lie)
            post_hour = int(float(split_line[2])//60)
            # print(post_hour)
            # print(e)

            time_lst = [hour for hour in range(post_hour+1) if hour<=61]
            write_path_lst = [root_path+'time_split/'+'Twitter15/'+str(hour)+'/' for hour in time_lst]
            write_path_lst = [j.rstrip('/') for j in write_path_lst]
            # [os.path.exists(k)
            for k in write_path_lst:
                isExists = os.path.exists(k)
                if not isExists:
                    os.makedirs(k)
                    print(k+' make success')
            #写入的时候应该考虑当前时间的前边所有的时间

            write_file_lst = [write_path+'/'+file_name+'.txt' for write_path in write_path_lst]
            # print(write_file_lst)
            # print(e)
            for write_file in write_file_lst:
                f = open(write_file,'a')
                f.write(line)
                f.close()

split_post_by_time(file_path)
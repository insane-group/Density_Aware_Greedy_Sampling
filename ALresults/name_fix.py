import sys
import os


if __name__ == '__main__':

    dir_path = sys.argv[1]
    
    for dataset in os.listdir(dir_path):

        print(dataset)

        for old_filename in os.listdir(os.path.join(dir_path, dataset)):
            print(old_filename)
            # old_filename_bits = old_filename.split("_")
            # dataset = old_filename_bits[1]
            # method  = old_filename_bits[-2]
            # number  = old_filename_bits[-1]

            # if dataset == "o2":
            #     dataset = "O2"

            # if dataset == "n2":
            #     dataset = "N2"

            # if dataset == "h2":
            #     dataset = "H2"

            # if dataset == "ch4":
            #     dataset = "CH4"

            # if dataset == "he":
            #     dataset = "He"

            # new_filename = f"{dataset}_150_{method}_{number}"


            new_filename = old_filename.replace('density','DAGS')
            # print(old_filename,new_filename)

            os.rename(os.path.join(dir_path, dataset, old_filename), os.path.join(dir_path, dataset, new_filename))

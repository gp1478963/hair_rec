import os
import pandas as pd



if __name__ == '__main__':
    data = pd.DataFrame()
    data_folder = './hair/data'
    sub_dir = os.scandir(data_folder)
    sub_dir = [entry.name for entry in sub_dir]
    aggregate = []
    data_type = []
    for index, sub_dir_name in enumerate(sub_dir):
        sub_dir_path = os.path.join(data_folder, sub_dir_name)
        class_data = os.scandir(sub_dir_path)
        for class_w in class_data:
            aggregate.append(os.path.join(data_folder, sub_dir_name, class_w.name))
            data_type.append(index)

    data['image'] = aggregate
    data['type'] = data_type
    data.to_csv('./data/hair_class.csv', index=False)
import csv
import os
import random
from shutil import copyfile


def get_age():
    gender_list = {}
    with open('voxceleb_gender.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            if row[2].strip() == 'm':
                gender_list.setdefault(row[0].strip(), 'male')
            else:
                gender_list.setdefault(row[0].strip(), 'female')

    voxceleb_list = {}
    con = 0
    with open('voxceleb_age.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            if row[3].strip() not in gender_list.keys():
                continue
            age = int(float(row[0]))
            gender = gender_list[row[3].strip()]
            voxceleb_list.setdefault(age, {})
            voxceleb_list[age].setdefault(gender, [])
            if os.path.exists(os.path.join('/mnt/disk1/chengxize/data/voxceleb_age', row[3].strip(), row[4])):
                voxceleb_list[age][gender].append(os.path.join(row[3].strip(), row[4]))
                con += 1

    return voxceleb_list


if __name__ == '__main__':
    get_age()

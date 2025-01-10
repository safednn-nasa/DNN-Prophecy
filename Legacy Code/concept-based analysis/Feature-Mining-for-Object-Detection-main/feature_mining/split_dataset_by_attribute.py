import sys
import argparse
import os
sys.path.insert(0, '/root/siqi/Feature-Mining-for-Object-Detection')

from nuimages import NuImages

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute-1', type=str, default='cycle.with_rider', help='the attribute you want to split')
    parser.add_argument('--attribute-2', type=str, default='cycle.with_rider', help='the attribute you want to split')
    parser.add_argument('--data-type', type=str, default='val', help='train or val')
    parser.add_argument('--nuimage-data', type=str, default='/root/data/nuimages/images/', help='nuimages path')
    parser.add_argument('--target-folder', type=str, default='/root/siqi/Feature-Mining-for-Object-Detection/split_result/', help='result path')
    opt = parser.parse_args()


    ###############  Load NuImages  ###############
    dataroot = opt.nuimage_data
    data_type = opt.data_type
    nuim = NuImages(dataroot=dataroot, version='v1.0-'+data_type, verbose=True, lazy=False)

    attribute_dict = {}
    for attri in nuim.attribute:
        attribute_dict[attri['token']] = attri['name']

    # Attributes for samples in training set
    data_attributes = {}
    for ob in nuim.object_ann:
        sample_token = ob['sample_data_token']

        # Get object's file name
        path = dataroot + nuim.get('sample_data', sample_token)['filename']
        if path not in data_attributes and ob['attribute_tokens']:
            data_attributes[path] = set()
        
        # Get object's attribute
        for attribute_token in ob['attribute_tokens']:
            data_attributes[path].add(attribute_dict[attribute_token])


    path_attri_1 = f'{opt.target_folder}with_{opt.attribute_1}.txt'
    path_attri_2 = f'{opt.target_folder}with_{opt.attribute_2}.txt'
    path_without = f'{opt.target_folder}other_{opt.attribute_1}_{opt.attribute_2}.txt'


    with open(path_attri_1, 'w') as f1, open(path_attri_2, 'w') as f2, open(path_without, 'w') as f3:

        for file in data_attributes.keys():
            if opt.attribute_1 in data_attributes[file] and opt.attribute_2 not in data_attributes[file]:
                f1.write(file + '\n')

            elif opt.attribute_1 not in data_attributes[file] and opt.attribute_2 in data_attributes[file]:
                f2.write(file + '\n')
            
            elif opt.attribute_1 not in data_attributes[file] and opt.attribute_2 not in data_attributes[file]:
                f3.write(file + '\n')

        f1.close()
        f2.close()
        f3.close()
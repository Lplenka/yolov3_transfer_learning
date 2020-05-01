import json
import logging
import os
import shutil

from pycocotools.coco import COCO

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('parso.python.diff').disabled = True

"""
Eample Usage

For Merging Annotations
In [1]: from annotations import Annotations                                                                                                                 

In [2]: ann_dir = "/home/linu/personal/Data/annotations_"                                                                                                               

In [3]: cas = Merge_Annotations(ann_dir=ann_dir)                                                                                                                        
                loading annotations into memory...
                Done (t=0.06s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.02s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.01s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.00s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.02s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.12s)
                creating index...
                index created!

In [4]: cas.merge()                                                                                                                                                     
Merging annotations
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 25.81it/s]

For Frequency Distribution
Eample Usage
In [1]: from annotations import Annotations                                                                                                                 

In [2]: ann_dir = "/home/linu/personal/Data/annotations_"                                                                                                               

In [3]: cas = Merge_Annotations(ann_dir=ann_dir)                                                                                                                        
                loading annotations into memory...
                Done (t=0.06s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.02s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.01s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.00s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.02s)
                creating index...
                index created!
                loading annotations into memory...
                Done (t=0.12s)
                creating index...
                index created!

In [4]: cas.show_frequency(save=False)                                                                                                                                                     
Merging annotations
100%|████████████████████████████████
"""


class Annotations():
    def __init__(self, ann_dir=None):
        """
        :param ann_dir (str): path to annotations folder.
        """
        self.ann_dir = ann_dir
        self.res_dir = os.path.join(os.path.dirname(ann_dir), 'results')

        # Create Results Directory
        if os.path.exists(self.res_dir) is False:
            os.mkdir(self.res_dir)

        self.jsonfiles = sorted(
            [j for j in os.listdir(ann_dir) if j[-5:] == ".json"])
        self.names = [n[:-5] for n in self.jsonfiles]

        # Note: Add check for confirming these folders only contain .jpg and .json respectively

        logging.debug("Number of annotation files = %s", len(self.jsonfiles))

        if not self.jsonfiles:
            raise AssertionError("Annotation files not passed")

        self.annfiles = [COCO(os.path.join(ann_dir, i))
                         for i in self.jsonfiles]
        self.anndict = dict(zip(self.jsonfiles, self.annfiles))

        self.ann_anchors = []

    def merge(self):
        """
        Function for merging multiple coco datasets
        """
        self.resann_dir = os.path.join(self.res_dir, 'merged', 'annotations')

        # Create directories for merged results and clear the previous ones
        # The exist_ok is for dealing with merged folder
        # TODO: Can be done better
        if os.path.exists(self.resann_dir) is False:
            os.makedirs(self.resann_dir, exist_ok=True)
        else:
            shutil.rmtree(self.resann_dir)
            os.makedirs(self.resann_dir, exist_ok=True)


        cann = {'images': [],
                'annotations': [],
                'info': None,
                'licenses': None,
                'categories': None}

        logging.debug("Merging Annotations...")

        dst_ann = os.path.join(self.resann_dir, 'merged.json')

        print("Merging annotations")
        for j in tqdm(self.jsonfiles):
            with open(os.path.join(self.ann_dir, j)) as a:
                cj = json.load(a)

            ind = self.jsonfiles.index(j)
            # Check if this is the 1st annotation.
            # If it is, continue else modify current annotation
            if ind == 0:
                cann['images'] = cann['images'] + cj['images']
                cann['annotations'] = cann['annotations'] + cj['annotations']
                if 'info' in list(cj.keys()):
                    cann['info'] = cj['info']
                if 'licenses' in list(cj.keys()):
                    cann['licenses'] = cj['licenses']
                cann['categories'] = cj['categories']

                last_imid = cann['images'][-1]['id']
                last_annid = cann['annotations'][-1]['id']

                # If last imid or last_annid is a str, convert it to int
                if isinstance(last_imid, str) or isinstance(last_annid, str):
                    logging.debug("String Ids detected. Converting to int")
                    id_dict = {}
                    # Change image id in images field
                    for i, im in enumerate(cann['images']):
                        id_dict[im['id']] = i
                        im['id'] = i

                    # Change annotation id & image id in annotations field
                    for i, im in enumerate(cann['annotations']):
                        im['id'] = i
                        if isinstance(last_imid, str):
                            im['image_id'] = id_dict[im['image_id']]

                last_imid = cann['images'][-1]['id']
                last_annid = cann['annotations'][-1]['id']

            else:

                id_dict = {}
                # Change image id in images field
                for i, im in enumerate(cj['images']):
                    id_dict[im['id']] = last_imid + i + 1
                    im['id'] = last_imid + i + 1

                # Change annotation and image ids in annotations field
                for i, ann in enumerate(cj['annotations']):
                    ann['id'] = last_annid + i + 1
                    ann['image_id'] = id_dict[ann['image_id']]

                cann['images'] = cann['images'] + cj['images']
                cann['annotations'] = cann['annotations'] + cj['annotations']
                if 'info' in list(cj.keys()):
                    cann['info'] = cj['info']
                if 'licenses' in list(cj.keys()):
                    cann['licenses'] = cj['licenses']
                cann['categories'] = cj['categories']

                last_imid = cann['images'][-1]['id']
                last_annid = cann['annotations'][-1]['id']

        with open(dst_ann, 'w') as aw:
            json.dump(cann, aw)


    def show_frequency(self, save=False):
        # Making axes iterable if only single annotation is present
        # Prepare annotations dataframe
        # This should be done at the start

        for ann, name in zip(self.annfiles, self.names):
            ann_df = pd.DataFrame(ann.anns).transpose()
            print("name:", name)
            print("hello", ann_df.columns)
            if len(ann_df) > 0:
                if 'category_id' in ann_df.columns:
                    #print(ann_df['category_id'].value_counts())
                    category_count = ann_df['category_id'].value_counts()
                    #category_count = category_count[:10, ]
                    plt.figure(figsize=(20, 10))
                    sns.barplot(category_count.index,
                                category_count.values, alpha=0.8)
                    plt.title('Frequency Bar plot')
                    plt.ylabel('Number of Occurrences', fontsize=12)
                    plt.xlabel('city', fontsize=12)
                    plt.show()
                    out_dir = os.path.join(os.getcwd(), 'results', 'plots')
                    if save is True:
                        if os.path.exists(out_dir) is False:
                            os.makedirs(out_dir)
                            plt.savefig(os.path.join(out_dir, name + "cat_dist"+ ".png"),
                                        bbox_inches='tight',
                                        pad_inches=1,
                                        dpi=plt.gcf().dpi)

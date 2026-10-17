"""Recreate the train/val split of COCO train2017 used by the paper's detection runs.

  python scripts/make_coco_split.py --coco <COCO root>
writes <COCO root>/annotations/coco2017_trn_split_anns.json and coco2017_val_split_anns.json from the stock
annotations/instances_train2017.json and the image-id lists in splits/. val2017 is the test set and is never used for
training or selection. (The two JSON files will also be published separately; see README.)"""
import argparse
import json
from pathlib import Path

SPLITS = Path(__file__).resolve().parents[1] / 'splits'


def subset(coco: dict, ids: set[int]) -> dict:
    return {**{k: v for k, v in coco.items() if k not in ('images', 'annotations')},
            'images': [im for im in coco['images'] if im['id'] in ids],
            'annotations': [a for a in coco['annotations'] if a['image_id'] in ids]}


def read_ids(name: str) -> set[int]:
    return {int(x) for x in (SPLITS / name).read_text().split()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coco', type=Path, required=True)
    a = ap.parse_args()
    ann = a.coco / 'annotations'
    coco = json.loads((ann / 'instances_train2017.json').read_text())
    trn_ids, val_ids = read_ids('coco2017_trn_split_image_ids.txt'), read_ids('coco2017_val_split_image_ids.txt')
    assert not trn_ids & val_ids, 'the two id lists overlap'
    missing = (trn_ids | val_ids) - {im['id'] for im in coco['images']}
    assert not missing, f'{len(missing)} listed images are not in instances_train2017.json'
    (ann / 'coco2017_trn_split_anns.json').write_text(json.dumps(subset(coco, trn_ids)))
    (ann / 'coco2017_val_split_anns.json').write_text(json.dumps(subset(coco, val_ids)))
    print(f'train: {len(trn_ids)} images, val: {len(val_ids)} images')


if __name__ == '__main__':
    main()

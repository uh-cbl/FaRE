import os
import csv
import argparse
from fare.io import Template, save_pkl


def load_csv_to_dict(csv_path):

    D = {}
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                TEMPLATE_ID, SUBJECT_ID, IM_PATH = int(line[0]), int(line[1]), line[2]
                base_d, base_f = IM_PATH.split('/')
                IM_PATH = os.path.join(base_d, '%s_%s' % (SUBJECT_ID, base_f))

                if TEMPLATE_ID in D:
                    if SUBJECT_ID in D[TEMPLATE_ID]:
                        D[TEMPLATE_ID][SUBJECT_ID].append(IM_PATH)
                    else:
                        D[TEMPLATE_ID] = {SUBJECT_ID: [IM_PATH]}
                else:
                    D[TEMPLATE_ID] = {SUBJECT_ID: [IM_PATH]}

    TEMPLATE_DICT = {}
    for TEMPLATE_ID, ITEM in D.items():
        SUBJECT_ID = list(ITEM.keys())[0]
        IM_PATHS = list(ITEM.values())[0]

        TEMPLATE_DICT[TEMPLATE_ID] = Template(IM_PATHS, SUBJECT_ID, TEMPLATE_ID)

    return TEMPLATE_DICT


def load_matches(csv_path):
    match_list = list()
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            id1, id2 = int(line[0]), int(line[1])
            match_list.append([id1, id2])

    return match_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('IJBAProtocol')
    args = parser.parse_args()

    IJBA_11_Matches = []
    # Verification
    Templates = {}

    for split in range(1, 11):
        metadata = load_csv_to_dict(os.path.join(args.IJBAProtocol, 'IJB-A_11_sets', 'split%d' % split,
                                                 'verify_metadata_%d.csv' % split))
        for template_id, item in metadata.items():
            if template_id not in Templates:
                Templates[template_id] = item

        match = load_matches(os.path.join(args.IJBAProtocol, 'IJB-A_11_sets', 'split%d' % split,
                                          'verify_comparisons_%d.csv' % split))
        IJBA_11_Matches.append(match)

    save_pkl('IJBA_11_Templates.pkl', Templates)
    save_pkl('IJBA_11_Matches.pkl', IJBA_11_Matches)
    # Identification
    IJBA_1N = []
    for split in range(1, 11):
        gallery = load_csv_to_dict(os.path.join(args.IJBAProtocol, 'IJB-A_1N_sets', 'split%d' % split,
                                                'search_gallery_%d.csv' % split))
        probe = load_csv_to_dict(os.path.join(args.IJBAProtocol, 'IJB-A_1N_sets', 'split%d' % split,
                                              'search_probe_%d.csv' % split))
        IJBA_1N.append({'gallery': gallery, 'probe': probe})

    save_pkl('IJBA_1N.pkl', IJBA_1N)

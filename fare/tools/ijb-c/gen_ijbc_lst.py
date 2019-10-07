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
    parser.add_argument('IJBCProtocol')

    args = parser.parse_args()

    G1 = load_csv_to_dict(os.path.join(args.IJBCProtocol, 'ijbc_1N_gallery_G1.csv'))
    G2 = load_csv_to_dict(os.path.join(args.IJBCProtocol, 'ijbc_1N_gallery_G2.csv'))
    Pr = load_csv_to_dict(os.path.join(args.IJBCProtocol, 'ijbc_1N_probe_mixed.csv'))
    Matches = load_matches(os.path.join(args.IJBCProtocol, 'ijbc_11_G1_G2_matches.csv'))

    # save path
    save_pkl('IJBC_1N_G1.pkl', G1)
    save_pkl('IJBC_1N_G2.pkl', G2)
    save_pkl('IJBC_1N_Pb.pkl', Pr)

    length = len(Matches) // 3
    Matches1, Matches2, Matches3 = Matches[: length], Matches[length: 2 * length], Matches[2*length:]
    save_pkl('IJBC_11_Matches-1.pkl', Matches1)
    save_pkl('IJBC_11_Matches-2.pkl', Matches2)
    save_pkl('IJBC_11_Matches-3.pkl', Matches3)

import os
from tqdm import tqdm
from ..datasets import LFW, FGLFW, CFP, UHDB31
from ..io import File, Template, Signature


def val_lfw(inference, db_dir, pre_processing, im_ext=None):
    dataset = LFW(db_dir)

    p_bar = tqdm(total=dataset._n_folds * (dataset.num_pos_pairs + dataset.num_neg_pairs))

    for i, fold in enumerate(dataset.im_folds):
        template_a_list, template_b_list = [], []
        for pair in fold['positives']:
            im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
                                                                                           pair['path2'])

            if im_ext is not None:
                im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
                im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)

            if os.path.exists(im_path1) and os.path.exists(im_path2):
                im1 = pre_processing(im_path1)
                im2 = pre_processing(im_path2)

                if im1 is not None and im2 is not None:
                    ft1 = inference(im1).asnumpy().flatten()
                    ft2 = inference(im2).asnumpy().flatten()
                    # Set template id of positives are 1
                    # Enroll the template
                    template_a = Template(template_id=1, subject_id=1, signature=ft1, list_im_path=[im_path1])
                    template_b = Template(template_id=1, subject_id=1, signature=ft2, list_im_path=[im_path2])

                    template_a_list.append(template_a)
                    template_b_list.append(template_b)

            p_bar.update()

        for pair in fold['negatives']:
            # Set template id of negatives are 0, 1
            im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
                                                                                           pair['path2'])
            if im_ext is not None:
                im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
                im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)

            if os.path.exists(im_path1) and os.path.exists(im_path2):
                im1 = pre_processing(im_path1)
                im2 = pre_processing(im_path2)

                if im1 is not None and im2 is not None:
                    ft1 = inference(im1).asnumpy().flatten()
                    ft2 = inference(im2).asnumpy().flatten()
                    # Set template id of positives are 1

                    template_a = Template(template_id=0, subject_id=0, signature=ft1, list_im_path=[im_path1])
                    template_b = Template(template_id=1, subject_id=1, signature=ft2, list_im_path=[im_path2])

                    template_a_list.append(template_a)
                    template_b_list.append(template_b)

            p_bar.update()

        dataset.template_folders[i]['template_a_list'] = template_a_list
        dataset.template_folders[i]['template_b_list'] = template_b_list

    p_bar.close()

    dataset.evaluate()
    dataset.compute_metrics()

    return dataset


def val_fglfw(inference, db_dir, pre_processing, im_ext=None):
    dataset = FGLFW(db_dir)

    p_bar = tqdm(total=dataset.num_folds * (dataset.num_pos_pairs + dataset.num_neg_pairs))

    for i, fold in enumerate(dataset.im_folds):
        template_a_list, template_b_list = [], []
        for pair in fold['positives']:
            im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
                                                                                           pair['path2'])

            if im_ext is not None:
                im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
                im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)

            if os.path.exists(im_path1) and os.path.exists(im_path2):
                im1 = pre_processing(im_path1)
                im2 = pre_processing(im_path2)

                if im1 is not None and im2 is not None:
                    ft1 = inference(im1).asnumpy().flatten()
                    ft2 = inference(im2).asnumpy().flatten()
                    # Set template id of positives are 1
                    # Enroll this file
                    template_a = Template(template_id=1, subject_id=1, signature=ft1, list_im_path=[im_path1])
                    template_b = Template(template_id=1, subject_id=1, signature=ft2, list_im_path=[im_path2])

                    template_a_list.append(template_a)
                    template_b_list.append(template_b)

            p_bar.update()

        for pair in fold['negatives']:
            # Set template id of negatives are 0, 1
            im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
                                                                                           pair['path2'])
            if im_ext is not None:
                im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
                im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)

            if os.path.exists(im_path1) and os.path.exists(im_path2):
                im1 = pre_processing(im_path1)
                im2 = pre_processing(im_path2)

                if im1 is not None and im2 is not None:
                    ft1 = inference(im1).asnumpy().flatten()
                    ft2 = inference(im2).asnumpy().flatten()
                    # Set template id of positives are 1
                    template_a = Template(template_id=0, subject_id=0, signature=ft1, list_im_path=[im_path1])
                    template_b = Template(template_id=1, subject_id=1, signature=ft2, list_im_path=[im_path2])

                    template_a_list.append(template_a)
                    template_b_list.append(template_b)

            p_bar.update()

        dataset.template_folders[i]['template_a_list'] = template_a_list
        dataset.template_folders[i]['template_b_list'] = template_b_list

    p_bar.close()

    dataset.evaluate()
    dataset.compute_metrics()

    return dataset


def val_cfp(inference, db_dir, pre_processing, im_ext=None, protocol='FP'):
    dataset = CFP(db_dir, db_protocol=protocol)

    p_bar = tqdm(total=dataset._n_folds * (dataset.num_pos_pairs + dataset.num_neg_pairs))

    for i, fold in enumerate(dataset.im_folds):
        template_a_list, template_b_list = [], []
        for pair in fold['positives']:
            im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
                                                                                           pair['path2'])

            if im_ext is not None:
                im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
                im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)

            if os.path.exists(im_path1) and os.path.exists(im_path2):
                im1 = pre_processing(im_path1)
                im2 = pre_processing(im_path2)

                if im1 is not None and im2 is not None:
                    ft1 = inference(im1).asnumpy().flatten()
                    ft2 = inference(im2).asnumpy().flatten()
                    # Set template id of positives are 1
                    # Enroll this file
                    template_a = Template(template_id=1, subject_id=1, signature=ft1, list_im_path=[im_path1])
                    template_b = Template(template_id=1, subject_id=1, signature=ft2, list_im_path=[im_path2])

                    template_a_list.append(template_a)
                    template_b_list.append(template_b)

            p_bar.update()

        for pair in fold['negatives']:
            # Set template id of negatives are 0, 1
            im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
                                                                                           pair['path2'])
            if im_ext is not None:
                im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
                im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)

            if os.path.exists(im_path1) and os.path.exists(im_path2):
                im1 = pre_processing(im_path1)
                im2 = pre_processing(im_path2)

                if im1 is not None and im2 is not None:
                    ft1 = inference(im1).asnumpy().flatten()
                    ft2 = inference(im2).asnumpy().flatten()
                    # Set template id of positives are 1
                    template_a = Template(template_id=0, subject_id=0, signature=ft1, list_im_path=[im_path1])
                    template_b = Template(template_id=1, subject_id=1, signature=ft2, list_im_path=[im_path2])

                    template_a_list.append(template_a)
                    template_b_list.append(template_b)

            p_bar.update()

        dataset.template_folders[i]['template_a_list'] = template_a_list
        dataset.template_folders[i]['template_b_list'] = template_b_list

    p_bar.close()

    dataset.evaluate()
    dataset.compute_metrics()

    return dataset


def val_uhdb31(inference, db_dir, pre_processing, im_ext=None):
    dataset = UHDB31(db_dir)

    p_bar = tqdm(total=len(dataset.im_folds))

    for i, fold in enumerate(dataset.im_folds):
        gallery_paths = fold['list_a']
        probe_paths = fold['list_b']

        # load gallery
        for im_path in gallery_paths:
            if im_ext is not None:
                im_path = im_path.replace(os.path.splitext(im_path)[1], im_ext)

            if os.path.exists(im_path):
                im = pre_processing(im_path)
                ft = inference(im).asnumpy().flatten()
                im_name = os.path.basename(im_path)
                template_id = int(im_name[:5])

                signature = Signature(ft)
                t = Template(template_id, template_id, signature=signature, list_im_path=[im_path])
                dataset.template_folders[i]['template_a_list'].append(t)
        # load probe
        for im_path in probe_paths:
            if im_ext is not None:
                im_path = im_path.replace(os.path.splitext(im_path)[1], im_ext)

            if os.path.exists(im_path):
                im = pre_processing(im_path)
                ft = inference(im).asnumpy().flatten()
                im_name = os.path.basename(im_path)
                template_id = int(im_name[:5])

                signature = Signature(ft)
                t = Template(template_id, template_id, signature=signature, list_im_path=[im_path])
                dataset.template_folders[i]['template_b_list'].append(t)

        p_bar.update()

    p_bar.close()

    dataset.evaluate()
    dataset.compute_metrics()

    return dataset


# def val_ijba_compare(inference, db_dir, pre_processing, im_ext=None):
#     dataset = IJB_A_Compare(db_dir)
#
#     p_bar = tqdm(total=dataset._n_folds)
#
#     for i, fold in enumerate(dataset.im_folds):
#         template_a_list, template_b_list = [], []
#         for pair in fold['positives']:
#             im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
#                                                                                            pair['path2'])
#
#             if im_ext is not None:
#                 im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
#                 im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)
#
#             if os.path.exists(im_path1) and os.path.exists(im_path2):
#                 ft1 = inference(pre_processing(im_path1)).asnumpy().flatten()
#                 ft2 = inference(pre_processing(im_path2)).asnumpy().flatten()
#                 # Set template id of positives are 1
#                 # Enroll this file
#                 file_a = File(os.path.join(dataset.db_dir, im_path1), template_id=1, subject_id=1, features=ft1)
#                 file_b = File(os.path.join(dataset.db_dir, im_path2), template_id=1, subject_id=1, features=ft2)
#                 # Enroll the template
#                 template_a = Template(template_id=1, subject_id=1, list_files=[file_a], method='mean')
#                 template_b = Template(template_id=1, subject_id=1, list_files=[file_b], method='mean')
#
#                 template_a_list.append(template_a)
#                 template_b_list.append(template_b)
#
#             p_bar.update()
#
#         for pair in fold['negatives']:
#             # Set template id of negatives are 0, 1
#             im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
#                                                                                            pair['path2'])
#             if im_ext is not None:
#                 im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
#                 im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)
#
#             if os.path.exists(im_path1) and os.path.exists(im_path2):
#                 ft1 = inference(pre_processing(im_path1)).asnumpy().flatten()
#                 ft2 = inference(pre_processing(im_path2)).asnumpy().flatten()
#                 # Set template id of positives are 1
#                 file_a = File(os.path.join(dataset.db_dir, im_path1), template_id=0, subject_id=0, feature_path=ft1)
#                 file_b = File(os.path.join(dataset.db_dir, im_path2), template_id=1, subject_id=1, feature_path=ft2)
#
#                 template_a = Template(template_id=0, subject_id=0, list_files=[file_a], method='mean')
#                 template_b = Template(template_id=1, subject_id=1, list_files=[file_b], method='mean')
#
#                 template_a_list.append(template_a)
#                 template_b_list.append(template_b)
#
#             p_bar.update()
#
#         dataset.template_folders[i]['template_a_list'] = template_a_list
#         dataset.template_folders[i]['template_b_list'] = template_b_list
#
#     p_bar.close()
#
#     dataset.evaluate()
#     dataset.compute_metrics()
#
#     return dataset
#
#
# def val_ijba_search(inference, db_dir, pre_processing, im_ext=None):
#     dataset = IJB_A_Search(db_dir)
#
#     p_bar = tqdm(total=dataset._n_folds * (dataset.num_pos_pairs + dataset.num_neg_pairs))
#
#     for i, fold in enumerate(dataset.im_folds):
#         template_a_list, template_b_list = [], []
#         for pair in fold['positives']:
#             im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
#                                                                                            pair['path2'])
#
#             if im_ext is not None:
#                 im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
#                 im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)
#
#             if os.path.exists(im_path1) and os.path.exists(im_path2):
#                 ft1 = inference(pre_processing(im_path1)).asnumpy().flatten()
#                 ft2 = inference(pre_processing(im_path2)).asnumpy().flatten()
#                 # Set template id of positives are 1
#                 # Enroll this file
#                 file_a = File(os.path.join(dataset.db_dir, im_path1), template_id=1, subject_id=1, features=ft1)
#                 file_b = File(os.path.join(dataset.db_dir, im_path2), template_id=1, subject_id=1, features=ft2)
#                 # Enroll the template
#                 template_a = Template(template_id=1, subject_id=1, list_files=[file_a], method='mean')
#                 template_b = Template(template_id=1, subject_id=1, list_files=[file_b], method='mean')
#
#                 template_a_list.append(template_a)
#                 template_b_list.append(template_b)
#
#             p_bar.update()
#
#         for pair in fold['negatives']:
#             # Set template id of negatives are 0, 1
#             im_path1, im_path2 = os.path.join(dataset.db_dir, pair['path1']), os.path.join(dataset.db_dir,
#                                                                                            pair['path2'])
#             if im_ext is not None:
#                 im_path1 = im_path1.replace(os.path.splitext(im_path1)[1], im_ext)
#                 im_path2 = im_path2.replace(os.path.splitext(im_path2)[1], im_ext)
#
#             if os.path.exists(im_path1) and os.path.exists(im_path2):
#                 ft1 = inference(pre_processing(im_path1)).asnumpy().flatten()
#                 ft2 = inference(pre_processing(im_path2)).asnumpy().flatten()
#                 # Set template id of positives are 1
#                 file_a = File(os.path.join(dataset.db_dir, im_path1), template_id=0, subject_id=0, feature_path=ft1)
#                 file_b = File(os.path.join(dataset.db_dir, im_path2), template_id=1, subject_id=1, feature_path=ft2)
#
#                 template_a = Template(template_id=0, subject_id=0, list_files=[file_a], method='mean')
#                 template_b = Template(template_id=1, subject_id=1, list_files=[file_b], method='mean')
#
#                 template_a_list.append(template_a)
#                 template_b_list.append(template_b)
#
#             p_bar.update()
#
#         dataset.template_folders[i]['template_a_list'] = template_a_list
#         dataset.template_folders[i]['template_b_list'] = template_b_list
#
#     p_bar.close()
#
#     dataset.evaluate()
#     dataset.compute_metrics()
#
#     return dataset

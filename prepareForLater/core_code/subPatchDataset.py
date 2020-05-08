import os, glob
import cv2
import sys
import numpy as np
import random
import imutils
from densenet.utils import processArguments, sortKey

params = {
    'db_root_dir': '/home/abhineet/N/Datasets/617/',
    'seq_name': 'training',
    'out_seq_name': '',
    'fname_templ': 'img',
    'img_ext': 'tif',
    'out_ext': 'png',
    'patch_height': 32,
    'patch_width': 0,
    'min_stride': 10,
    'max_stride': 0,
    'enable_flip': 0,
    'enable_rot': 0,
    'min_rot': 10,
    'max_rot': 0,
    'show_img': 0,
    'n_frames': 0,
    'start_id': 0,
    'end_id': -1,
    'enable_labels': 1
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)
    db_root_dir = params['db_root_dir']
    seq_name = params['seq_name']
    out_seq_name = params['out_seq_name']
    fname_templ = params['fname_templ']
    img_ext = params['img_ext']
    out_ext = params['out_ext']
    show_img = params['show_img']
    _patch_height = params['patch_height']
    _patch_width = params['patch_width']
    min_stride = params['min_stride']
    max_stride = params['max_stride']
    enable_flip = params['enable_flip']
    enable_rot = params['enable_rot']
    min_rot = params['min_rot']
    max_rot = params['max_rot']
    n_frames = params['n_frames']
    start_id = params['start_id']
    end_id = params['end_id']
    enable_labels = params['enable_labels']

    if enable_labels:
        labels_path = os.path.join(db_root_dir, seq_name, 'labels')
        if not os.path.isdir(labels_path):
            print('Labels folder does not exist so disabling it')
            enable_labels = 0

    if enable_rot and not enable_labels:
        raise SystemError('Rotation cannot be enabled without labels')

    src_path = os.path.join(db_root_dir, seq_name, 'images')
    print('Reading source images from: {}'.format(src_path))

    src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(img_ext))]
    total_frames = len(src_files)
    if total_frames <= 0:
        raise SystemError('No input frames found')
    print('total_frames: {}'.format(total_frames))
    src_files.sort(key=sortKey)
    # src_file_list = src_file_list.sort()

    if n_frames <= 0:
        n_frames = total_frames

    if end_id < start_id:
        end_id = n_frames - 1

    patch_width, patch_height = _patch_width, _patch_height

    if patch_width <= 0:
        patch_width = patch_height

    if min_stride <= 0:
        min_stride = patch_height

    if max_stride <= min_stride:
        max_stride = min_stride

    image_as_patch = 0
    if _patch_width <= 0 and _patch_height <= 0:
        image_as_patch = 1
        print('Using entire image as the patch')

    if not out_seq_name:
        if image_as_patch:
            out_seq_name = '{:s}_{:d}_{:d}'.format(seq_name, start_id, end_id)
        else:
            out_seq_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride)

        if enable_rot:
            out_seq_name = '{}_rot_{:d}_{:d}'.format(out_seq_name, min_rot, max_rot)

        if enable_flip:
            out_seq_name = '{}_flip'.format(out_seq_name)

    print('Writing output images to: {}'.format(out_seq_name))

    out_src_path = os.path.join(db_root_dir, out_seq_name, 'images')
    if not os.path.isdir(out_src_path):
        os.makedirs(out_src_path)

    if enable_labels:
        out_labels_path = os.path.join(db_root_dir, out_seq_name, 'labels')
        if not os.path.isdir(out_labels_path):
            os.makedirs(out_labels_path)

    rot_angle = 0
    n_out_frames = 0

    pause_after_frame = 1

    n_frames = end_id - start_id + 1

    for img_id in range(start_id, end_id + 1):

        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
        img_fname = src_files[img_id]
        img_fname_no_ext = os.path.splitext(img_fname)[0]

        src_img_fname = os.path.join(src_path, img_fname)
        src_img = cv2.imread(src_img_fname)

        src_height, src_width, _ = src_img.shape

        if image_as_patch:
            patch_width, patch_height = src_width, src_height

        if src_height < patch_height or src_width < patch_width:
            print('\nImage {} is too small {}x{} for the given patch size {}x{}\n'.format(
                src_img_fname, src_width, src_height, patch_width, patch_height))
            continue

        if src_img is None:
            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

        if enable_labels:
            labels_img_fname = os.path.join(labels_path, img_fname)
            labels_img = cv2.imread(labels_img_fname)
            labels_img_orig = np.copy(labels_img)

            if labels_img is None:
                raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))

        if enable_rot:
            labels_img[labels_img < 64] = 50
            labels_img[np.logical_and(labels_img >= 64, labels_img < 192)] = 150
            labels_img[labels_img >= 192] = 250

            rot_angle = random.randint(min_rot, max_rot)
            src_img = imutils.rotate_bound(src_img, rot_angle)
            labels_img = imutils.rotate_bound(labels_img, rot_angle).astype(np.int32)

            out_src_img_fname = os.path.join(db_root_dir, out_seq_name, 'img_{:d}_rot_{:d}.{:s}'.format(
                img_id + 1, rot_angle, out_ext))

            out_labels_img_fname = os.path.join(db_root_dir, out_seq_name, 'labels_{:d}_rot_{:d}.{:s}'.format(
                img_id + 1, rot_angle, out_ext))

            cv2.imwrite(out_src_img_fname, src_img)
            cv2.imwrite(out_labels_img_fname, labels_img)

            labels_img[labels_img == 0] = -1

        n_rows, ncols, n_channels = src_img.shape

        if enable_labels:
            _n_rows, _ncols, _n_channels = labels_img.shape

            if n_rows != _n_rows or n_rows != _n_rows or n_rows != _n_rows:
                raise SystemError('Dimension mismatch between image and labels for file: {}'.format(img_fname))

            labels_img[np.logical_and(labels_img >= 0, labels_img < 64)] = 0
            labels_img[np.logical_and(labels_img >= 64, labels_img < 192)] = 1
            labels_img[labels_img >= 192] = 2

        # np.savetxt(os.path.join(db_root_dir, seq_name, 'labels_img_{}.txt'.format(img_id + 1)),
        #            labels_img[:, :, 2], fmt='%d')

        out_id = 0
        # skip_id = 0
        min_row = 0

        while True:
            max_row = min_row + patch_height
            if max_row > n_rows:
                diff = max_row - n_rows
                min_row -= diff
                max_row -= diff

            min_col = 0
            while True:
                max_col = min_col + patch_width
                if max_col > ncols:
                    diff = max_col - ncols
                    min_col -= diff
                    max_col -= diff

                if enable_labels:
                    labels_patch = labels_img[min_row:max_row, min_col:max_col, :]

                if enable_rot and (labels_patch == -1).any():
                    pass
                    # skip_id += 1
                    # sys.stdout.write('\rSkipping patch {:d}'.format(skip_id))
                else:
                    src_patch = src_img[min_row:max_row, min_col:max_col, :]

                    if image_as_patch:
                        out_img_fname = img_fname_no_ext
                    else:
                        out_img_fname = '{:s}_{:d}'.format(img_fname_no_ext, out_id + 1)

                    if enable_rot:
                        labels_patch = labels_patch.astype(np.uint8)
                        out_img_fname = '{:s}_rot_{:d}'.format(out_img_fname, rot_angle)

                    out_id += 1

                    out_src_img_fname = os.path.join(out_src_path, '{:s}.{:s}'.format(out_img_fname, out_ext))
                    cv2.imwrite(out_src_img_fname, src_patch)
                    if show_img:
                        disp_img = src_img.copy()
                        cv2.rectangle(disp_img, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)

                        # disp_labels_img = labels_img.copy()
                        # cv2.rectangle(disp_labels_img, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)

                        cv2.imshow('src_img', disp_img)
                        cv2.imshow('patch', src_patch)
                        if enable_labels:
                            labels_patch_disp = labels_img_orig[min_row:max_row, min_col:max_col, :].astype(np.uint8)
                            cv2.imshow('patch_labels', labels_patch_disp)
                        # cv2.imshow('disp_labels_img', disp_labels_img)
                        k = cv2.waitKey(1 - pause_after_frame)
                        if k == 27:
                            sys.exit(0)
                        elif k == 32:
                            pause_after_frame = 1 - pause_after_frame
                    if enable_labels:
                        out_labels_img_fname = os.path.join(out_labels_path, '{:s}.{:s}'.format(out_img_fname, out_ext))
                        cv2.imwrite(out_labels_img_fname, labels_patch)

                    if enable_flip:
                        src_patch_lr = np.fliplr(src_patch)
                        out_src_img_fname = os.path.join(out_src_path, '{:s}_lr.{:s}'.format(out_img_fname, out_ext))
                        cv2.imwrite(out_src_img_fname, src_patch_lr)

                        src_patch_ud = np.flipud(src_patch)
                        out_src_img_fname = os.path.join(out_src_path, '{:s}_ud.{:s}'.format(out_img_fname, out_ext))
                        cv2.imwrite(out_src_img_fname, src_patch_ud)

                        if enable_labels:
                            labels_patch_lr = np.fliplr(labels_patch)
                            out_labels_img_fname = os.path.join(out_labels_path,
                                                                '{:s}_lr.{:s}'.format(out_img_fname, out_ext))
                            cv2.imwrite(out_labels_img_fname, labels_patch_lr)

                            labels_patch_ud = np.flipud(labels_patch)
                            out_labels_img_fname = os.path.join(out_labels_path,
                                                                '{:s}_ud.{:s}'.format(out_img_fname, out_ext))
                            cv2.imwrite(out_labels_img_fname, labels_patch_ud)
                min_col += random.randint(min_stride, max_stride)
                if image_as_patch or max_col >= ncols:
                    break

            if image_as_patch or max_row >= n_rows:
                break
            min_row += random.randint(min_stride, max_stride)


        # sys.stdout.write('\nDone {:d}/{:d} frames with {:d} patches\n'.format(
        #     img_id + 1, n_frames, out_id))
        # sys.stdout.flush()
        n_out_frames += out_id
        sys.stdout.write('\rDone {:d}/{:d} frames'.format(img_id + 1, n_frames))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
    sys.stdout.write('Total frames generated: {}\n'.format(n_out_frames))

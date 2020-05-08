import sys, os
import subprocess
from densenet.utils import processArguments

params = {
    'db_root_dir': '/home/abhineet/N/Datasets/617/',
    'seq_name': 'training',
    'fname_templ': 'img',
    'img_ext': 'tif',
    'out_ext': 'png',
    'patch_height': 32,
    'patch_width': 0,
    'min_stride': 10,
    'max_stride': 0,
    'enable_flip': 0,
    'min_rot': 15,
    'max_rot': 345,
    'n_rot': 3,
    'show_img': 0,
    'n_frames': 0,
    'start_id': 0,
    'end_id': -1,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)
    db_root_dir = params['db_root_dir']
    seq_name = params['seq_name']
    fname_templ = params['fname_templ']
    img_ext = params['img_ext']
    out_ext = params['out_ext']
    show_img = params['show_img']
    patch_height = params['patch_height']
    patch_width = params['patch_width']
    min_stride = params['min_stride']
    max_stride = params['max_stride']
    enable_flip = params['enable_flip']
    min_rot = params['min_rot']
    max_rot = params['max_rot']
    n_rot = params['n_rot']
    n_frames = params['n_frames']
    start_id = params['start_id']
    end_id = params['end_id']

    src_path = os.path.join(db_root_dir, seq_name, 'images')
    src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(img_ext))]
    total_frames = len(src_files)
    # print('file_list: {}'.format(file_list))
    if total_frames <= 0:
        raise SystemError('No input frames found')
    print('total_frames: {}'.format(total_frames))

    if n_frames <= 0:
        n_frames = total_frames

    if end_id < start_id:
        end_id = n_frames - 1

    if patch_width <= 0:
        patch_width = patch_height

    if min_stride <= 0:
        min_stride = patch_height

    if max_stride <= min_stride:
        max_stride = min_stride

    rot_range = int(float(max_rot - min_rot) / float(n_rot))

    out_seq_names = []

    cmb_out_seq_name = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_rot_{:d}_{:d}_{:d}'.format(
        seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride, min_rot, max_rot, n_rot)

    if enable_flip:
        cmb_out_seq_name = '{}_flip'.format(cmb_out_seq_name)

    base_cmd = 'python3 subPatchDataset.py db_root_dir={} seq_name={} fname_templ={} img_ext={} out_ext={} ' \
               'patch_height={} patch_width={} min_stride={} max_stride={} enable_flip={} start_id={} end_id={} ' \
               'n_frames={} show_img={} out_seq_name={}'.format(
        db_root_dir, seq_name, fname_templ, img_ext, out_ext, patch_height, patch_width, min_stride, max_stride,
        enable_flip, start_id, end_id, n_frames, show_img, cmb_out_seq_name)

    # out_seq_name_base = '{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
    #     seq_name, start_id, end_id, patch_height, patch_width, min_stride, max_stride)
    # out_seq_name = out_seq_name_base
    # if enable_flip:
    #     out_seq_name = '{}_flip'.format(out_seq_name_base)
    # out_seq_names.append(out_seq_name)

    no_rot_cmd = '{} enable_rot=0'.format(base_cmd)
    print('\n\nRunning:\n {}\n\n'.format(no_rot_cmd))
    subprocess.check_call(no_rot_cmd, shell=True)

    _min_rot = min_rot

    for i in range(n_rot):
        if i == n_rot - 1:
            _max_rot = max_rot
        else:
            _max_rot = _min_rot + rot_range

        rot_cmd = '{} enable_rot=1 min_rot={} max_rot={}'.format(base_cmd, _min_rot, _max_rot)
        print('\n\nRunning:\n {}\n\n'.format(rot_cmd))

        subprocess.check_call(rot_cmd, shell=True)

        # out_seq_name = '{}_rot_{:d}_{:d}'.format(out_seq_name_base, min_rot, max_rot)
        # if enable_flip:
        #     out_seq_name = '{}_flip'.format(out_seq_name_base)
        # out_seq_names.append(out_seq_name)

        # cmb_out_seq_name_base = '{}_{}'.format(cmb_out_seq_name_base, _min_rot)
        _min_rot = _max_rot + 1

        # merge_base_cmb = 'python3 mergeDatasets.py'
        # for i in range(len(out_seq_names)):
        #     merge_cmd = '{} {} {} start_id={} end_id={}'.format(
        #         merge_base_cmb, out_seq_names[i], cmb_out_seq_name, start_id, end_id)
        #     print('\n\nRunning:\n {}\n\n'.format(merge_cmd))
        #     subprocess.check_call(merge_cmd, shell=True)



"""
Procedure for unit-testing with images:

1. Run unit tests at least once; this initializes a git clone of
   vispy/test-data in config['test_data_path']. This path is
   `~/.vispy/test-data` unless the config variable has been modified.
   The config file is located at `vispy/vispy/util/config.py`

2. Run individual test scripts with the --vispy-audit flag:

       $ python vispy/visuals/tests/test_ellipse.py --vispy-audit

   Any failing tests will
   display the test results, standard image, and the differences between the
   two. If the test result is bad, then press (f)ail. If the test result is
   good, then press (p)ass and the new image will be saved to the test-data
   directory.

3. After adding or changing test images, create a new commit:

        $ cd ~/.vispy/test-data
        $ git add ...
        $ git commit -a

4. Look up the most recent tag name from the `test_data_tag` variable in
   get_test_data_repo() below. Increment the tag name by 1 in the function
   and create a new tag in the test-data repository:

        $ git tag test-data-NNN
        $ git push --tags origin master

    This tag is used to ensure that each vispy commit is linked to a specific
    commit in the test-data repository. This makes it possible to push new
    commits to the test-data repository without interfering with existing
    tests, and also allows unit tests to continue working on older vispy
    versions.

    Finally, update the tag name in ``get_test_data_repo`` to the new name.

"""

import time
import os
import sys
import inspect
import base64
from subprocess import check_call, CalledProcessError
import numpy as np

from ..ext.six import string_types
from ..ext.six.moves import http_client as httplib
from ..ext.six.moves import urllib_parse as urllib
from .. import scene, config
from ..io import read_png, write_png
from ..gloo.util import _screenshot
from ..util import run_subprocess


tester = None


def _get_tester():
    global tester
    if tester is None:
        tester = ImageTester()
    return tester


def assert_image_approved(image, standard_file, message=None, **kwargs):
    """Check that an image test result matches a pre-approved standard.

    If the result does not match, then the user can optionally invoke a GUI
    to compare the images and decide whether to fail the test or save the new
    image as the standard.

    This function will automatically clone the test-data repository into
    ~/.vispy/test-data. However, it is up to the user to ensure this repository
    is kept up to date and to commit/push new images after they are saved.

    Run the test with python <test-path> --vispy-audit-tests to bring up
    the auditing GUI.

    Parameters
    ----------
    image : (h, w, 4) ndarray or 'screenshot'
        The test result to check
    standard_file : str
        The name of the approved test image to check against. This file name
        is relative to the root of the vispy test-data repository and will
        be automatically fetched.
    message : str
        A string description of the image. It is recommended to describe
        specific features that an auditor should look for when deciding whether
        to fail a test.

    Extra keyword arguments are used to set the thresholds for automatic image
    comparison (see ``assert_image_match()``).
    """

    if isinstance(image, string_types) and image == "screenshot":
        image = _screenshot(alpha=True)
    if message is None:
        code = inspect.currentframe().f_back.f_code
        message = "%s::%s" % (code.co_filename, code.co_name)

    # Make sure we have a test data repo available, possibly invoking git
    data_path = get_test_data_repo()

    # Read the standard image if it exists
    std_file = os.path.join(data_path, standard_file)
    if not os.path.isfile(std_file):
        std_image = None
    else:
        std_image = read_png(std_file)

    # If the test image does not match, then we go to audit if requested.
    try:
        if image.shape != std_image.shape:
            # Allow im1 to be an integer multiple larger than im2 to account
            # for high-resolution displays
            ims1 = np.array(image.shape).astype(float)
            ims2 = np.array(std_image.shape).astype(float)
            sr = ims1 / ims2
            if (sr[0] != sr[1] or not np.allclose(sr, np.round(sr)) or
               sr[0] < 1):
                raise TypeError("Test result shape %s is not an integer factor"
                                " larger than standard image shape %s." %
                                (ims1, ims2))
            sr = np.round(sr).astype(int)
            image = downsample(image, sr[0], axis=(0, 1)).astype(image.dtype)

        assert_image_match(image, std_image, **kwargs)
    except Exception:
        if standard_file in git_status(data_path):
            print("\n\nWARNING: unit test failed against modified standard "
                  "image %s.\nTo revert this file, run `cd %s; git checkout "
                  "%s`\n" % (std_file, data_path, standard_file))
        if config['audit_tests']:
            sys.excepthook(*sys.exc_info())
            _get_tester().test(image, std_image, message)
            std_path = os.path.dirname(std_file)
            print('Saving new standard image to "%s"' % std_file)
            if not os.path.isdir(std_path):
                os.makedirs(std_path)
            write_png(std_file, image)
        else:
            if std_image is None:
                raise Exception("Test standard %s does not exist." % std_file)
            else:
                if os.getenv('TRAVIS') is not None or \
                        os.getenv('APPVEYOR') is not None:
                    _save_failed_test(image, std_image, standard_file)
                raise


def assert_image_match(im1, im2, min_corr=0.9, px_threshold=50.,
                       px_count=None, max_px_diff=None, avg_px_diff=None,
                       img_diff=None):
    """Check that two images match.

    Images that differ in shape or dtype will fail unconditionally.
    Further tests for similarity depend on the arguments supplied.

    Parameters
    ----------
    im1 : (h, w, 4) ndarray
        Test output image
    im2 : (h, w, 4) ndarray
        Test standard image
    min_corr : float or None
        Minimum allowed correlation coefficient between corresponding image
        values (see numpy.corrcoef)
    px_threshold : float
        Minimum value difference at which two pixels are considered different
    px_count : int or None
        Maximum number of pixels that may differ
    max_px_diff : float or None
        Maximum allowed difference between pixels
    avg_px_diff : float or None
        Average allowed difference between pixels
    img_diff : float or None
        Maximum allowed summed difference between images

    """
    assert im1.ndim == 3
    assert im1.shape[2] == 4
    assert im1.dtype == im2.dtype

    diff = im1.astype(float) - im2.astype(float)
    if img_diff is not None:
        assert np.abs(diff).sum() <= img_diff

    pxdiff = diff.max(axis=2)  # largest value difference per pixel
    mask = np.abs(pxdiff) >= px_threshold
    if px_count is not None:
        assert mask.sum() <= px_count

    masked_diff = diff[mask]
    if max_px_diff is not None and masked_diff.size > 0:
        assert masked_diff.max() <= max_px_diff
    if avg_px_diff is not None and masked_diff.size > 0:
        assert masked_diff.mean() <= avg_px_diff

    if min_corr is not None:
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(im1.ravel(), im2.ravel())[0, 1]
        assert corr >= min_corr


def _save_failed_test(data, expect, filename):
    from ..io import _make_png
    commit, error = run_subprocess(['git', 'rev-parse',  'HEAD'])
    name = filename.split('/')
    name.insert(-1, commit.strip())
    filename = '/'.join(name)
    host = 'data.vispy.org'

    # concatenate data, expect, and diff into a single image
    ds = data.shape
    es = expect.shape

    shape = (max(ds[0], es[0]) + 4, ds[1] + es[1] + 8 + max(ds[1], es[1]), 4)
    img = np.empty(shape, dtype=np.ubyte)
    img[..., :3] = 100
    img[..., 3] = 255

    img[2:2+ds[0], 2:2+ds[1], :ds[2]] = data
    img[2:2+es[0], ds[1]+4:ds[1]+4+es[1], :es[2]] = expect

    diff = make_diff_image(data, expect)
    img[2:2+diff.shape[0], -diff.shape[1]-2:-2] = diff

    png = _make_png(img)
    conn = httplib.HTTPConnection(host)
    req = urllib.urlencode({'name': filename,
                            'data': base64.b64encode(png)})
    conn.request('POST', '/upload.py', req)
    response = conn.getresponse().read()
    conn.close()
    print("\nImage comparison failed. Test result: %s %s   Expected result: "
          "%s %s" % (data.shape, data.dtype, expect.shape, expect.dtype))
    print("Uploaded to: \nhttp://%s/data/%s" % (host, filename))
    if not response.startswith(b'OK'):
        print("WARNING: Error uploading data to %s" % host)
        print(response)


def make_diff_image(im1, im2):
    """Return image array showing the differences between im1 and im2.

    Handles images of different shape. Alpha channels are not compared.
    """
    ds = im1.shape
    es = im2.shape

    diff = np.empty((max(ds[0], es[0]), max(ds[1], es[1]), 4), dtype=int)
    diff[..., :3] = 128
    diff[..., 3] = 255
    diff[:ds[0], :ds[1], :min(ds[2], 3)] += im1[..., :3]
    diff[:es[0], :es[1], :min(es[2], 3)] -= im2[..., :3]
    diff = np.clip(diff, 0, 255).astype(np.ubyte)
    return diff


def downsample(data, n, axis=0):
    """Downsample by averaging points together across axis.
    If multiple axes are specified, runs once per axis.
    """
    if hasattr(axis, '__len__'):
        if not hasattr(n, '__len__'):
            n = [n]*len(axis)
        for i in range(len(axis)):
            data = downsample(data, n[i], axis[i])
        return data

    if n <= 1:
        return data
    nPts = int(data.shape[axis] / n)
    s = list(data.shape)
    s[axis] = nPts
    s.insert(axis+1, n)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(0, nPts*n)
    d1 = data[tuple(sl)]
    d1.shape = tuple(s)
    d2 = d1.mean(axis+1)

    return d2


class ImageTester(scene.SceneCanvas):
    """Graphical interface for auditing image comparison tests.
    """
    def __init__(self):
        self.grid = None
        self.views = None
        self.console = None
        self.last_key = None

        scene.SceneCanvas.__init__(self, size=(1000, 800))

        self.bgcolor = (0.1, 0.1, 0.1, 1)
        self.grid = self.central_widget.add_grid()
        border = (0.3, 0.3, 0.3, 1)
        self.views = (self.grid.add_view(row=0, col=0, border_color=border),
                      self.grid.add_view(row=0, col=1, border_color=border),
                      self.grid.add_view(row=0, col=2, border_color=border))
        label_text = ['test output', 'standard', 'diff']
        for i, v in enumerate(self.views):
            v.camera = 'panzoom'
            v.camera.aspect = 1
            v.camera.flip = (False, True)
            # unfreeze it to set the image and label on the view
            # this is slightly hacky, but it is simpler than
            # creating another class/storing as a dict or a tuple
            v.unfreeze()
            v.image = scene.Image(parent=v.scene)
            v.label = scene.Text(label_text[i], parent=v, color='yellow',
                                 anchor_x='left', anchor_y='top')
            v.freeze()

        self.views[1].camera.link(self.views[0].camera)
        self.views[2].camera.link(self.views[0].camera)
        self.console = scene.Console(text_color='white', border_color=border)
        self.grid.add_widget(self.console, row=1, col=0, col_span=3)

    def test(self, im1, im2, message):
        self.show()
        self.console.write('------------------')
        self.console.write(message)
        if im2 is None:
            self.console.write('Image1: %s %s   Image2: [no standard]' %
                               (im1.shape, im1.dtype))
            im2 = np.zeros((1, 1, 3), dtype=np.ubyte)
        else:
            self.console.write('Image1: %s %s   Image2: %s %s' %
                               (im1.shape, im1.dtype, im2.shape, im2.dtype))
        self.console.write('(P)ass or (F)ail this test?')
        self.views[0].image.set_data(im1)
        self.views[1].image.set_data(im2)
        diff = make_diff_image(im1, im2)

        self.views[2].image.set_data(diff)
        self.views[0].camera.set_range()

        while True:
            self.app.process_events()
            if self.last_key is None:
                pass
            elif self.last_key.lower() == 'p':
                self.console.write('PASS')
                break
            elif self.last_key.lower() in ('f', 'esc'):
                self.console.write('FAIL')
                raise Exception("User rejected test result.")
            time.sleep(0.03)

        for v in self.views:
            v.image.set_data(np.zeros((1, 1, 3), dtype=np.ubyte))

    def on_key_press(self, event):
        self.last_key = event.key.name


def get_test_data_repo():
    """Return the path to a git repository with the required commit checked
    out.

    If the repository does not exist, then it is cloned from
    https://github.com/vispy/test-data. If the repository already exists
    then the required commit is checked out.
    """

    # This tag marks the test-data commit that this version of vispy should
    # be tested against. When adding or changing test images, create
    # and push a new tag and update this variable.
    test_data_tag = 'test-data-7'

    data_path = config['test_data_path']
    git_path = 'http://github.com/vispy/test-data'
    gitbase = git_cmd_base(data_path)

    if os.path.isdir(data_path):
        # Already have a test-data repository to work with.

        # Get the commit ID of test_data_tag. Do a fetch if necessary.
        try:
            tag_commit = git_commit_id(data_path, test_data_tag)
        except NameError:
            cmd = gitbase + ['fetch', '--tags', 'origin']
            print(' '.join(cmd))
            check_call(cmd)
            try:
                tag_commit = git_commit_id(data_path, test_data_tag)
            except NameError:
                raise Exception("Could not find tag '%s' in test-data repo at"
                                " %s" % (test_data_tag, data_path))
        except Exception:
            if not os.path.exists(os.path.join(data_path, '.git')):
                raise Exception("Directory '%s' does not appear to be a git "
                                "repository. Please remove this directory." %
                                data_path)
            else:
                raise

        # If HEAD is not the correct commit, then do a checkout
        if git_commit_id(data_path, 'HEAD') != tag_commit:
            print("Checking out test-data tag '%s'" % test_data_tag)
            check_call(gitbase + ['checkout', test_data_tag])

    else:
        print("Attempting to create git clone of test data repo in %s.." %
              data_path)

        parent_path = os.path.split(data_path)[0]
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)

        if os.getenv('TRAVIS') is not None:
            # Create a shallow clone of the test-data repository (to avoid
            # downloading more data than is necessary)
            os.makedirs(data_path)
            cmds = [
                gitbase + ['init'],
                gitbase + ['remote', 'add', 'origin', git_path],
                gitbase + ['fetch', '--tags', 'origin', test_data_tag,
                           '--depth=1'],
                gitbase + ['checkout', '-b', 'master', 'FETCH_HEAD'],
            ]
        else:
            # Create a full clone
            cmds = [['git', 'clone', git_path, data_path]]

        for cmd in cmds:
            print(' '.join(cmd))
            rval = check_call(cmd)
            if rval == 0:
                continue
            raise RuntimeError("Test data path '%s' does not exist and could "
                               "not be created with git. Either create a git "
                               "clone of %s or set the test_data_path "
                               "variable to an existing clone." %
                               (data_path, git_path))

    return data_path


def git_cmd_base(path):
    return ['git', '--git-dir=%s/.git' % path, '--work-tree=%s' % path]


def git_status(path):
    """Return a string listing all changes to the working tree in a git
    repository.
    """
    cmd = git_cmd_base(path) + ['status', '--porcelain']
    return run_subprocess(cmd, stderr=None, universal_newlines=True)[0]


def git_commit_id(path, ref):
    """Return the commit id of *ref* in the git repository at *path*.
    """
    cmd = git_cmd_base(path) + ['show', ref]
    try:
        output = run_subprocess(cmd, stderr=None, universal_newlines=True)[0]
    except CalledProcessError:
        raise NameError("Unknown git reference '%s'" % ref)
    commit = output.split('\n')[0]
    assert commit[:7] == 'commit '
    return commit[7:]

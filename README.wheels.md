# Building and uploading wheels

## For OSX

We automate OSX wheel building using a custom github repository that builds on
the travis-ci OSX machines.

The travis-ci interface for the builds is :
https://travis-ci.org/MacPython/netcdf4-python-wheels

The driving github repository is :
https://github.com/MacPython/netcdf4-python-wheels

### How it works

The wheel-building repository:

* does a fresh build of the required C / C++ libraries;
* builds a netcdf4-python wheel, linking against these fresh builds;
* processes the wheel using [delocate](https://pypi.python.org/pypi/delocate).
  `delocate` copies the required dynamic libraries into the wheel and relinks
  the extension modules against the copied libraries;
* uploads the built wheel to http://wheels.scipy.org (a Rackspace container
  kindly donated by Rackspace to scikit-learn).

The resulting wheel is therefore self-contained and does not need any external
dynamic libraries apart from those provided as standard by OSX.

### Triggering a build

You will need write permission to the github repository to trigger new builds
on the travis-ci interface.  Contact us on the mailing list if you need this.

You can trigger a build by:

* making a commit to the `netcdf4-python-wheels` repository (e.g. with `git
  commit --allow-empty`); or
* clicking on the circular arrow icon towards the top right of the travis-ci
  page, to rerun the previous build.

In general, it is better to trigger a build with a commit, because this makes
a new set of build products and logs, keeping the old ones for reference.
Keeping the old build logs helps us keep track of previous problems and
successful builds.

### Which netcdf4-python commit does the repository build?

By default, the `netcd4-python-wheels` repository is usually set up to build
the latest git tag.  To check whether this is so have a look around line 5 of
`.travis.yml` in the `netcdf4-python-wheels` repository.  You should see
something like:

```
- BUILD_COMMIT='latest-tag'
```

If this is commented out, then the repository is set up to build the current
commit in the `netcdf4-python` submodule of the repository.  If it is set to
another value then it will be specifying a commit to build.

You can therefore build any arbitrary commit by specificying the commit hash
or branch name or tag name in this line of the `.travis.yml` file.

### Uploading the built wheels to pypi

Be careful, http://wheels.scipy.org points to a container on a distributed
content delivery network.  It can take up to 15 minutes for the new wheel file
to get updated into the container at http://wheels.scipy.org.

When the wheels are updated, you can of course just download them to your
machine manually, and then upload them manually to pypi, or by using
[twine][twine].  You can also use a script for doing this, housed at :
https://github.com/MacPython/terryfy/blob/master/wheel-uploader

You'll need [twine][twine] and [beautiful soup 4][bs4].

You will typically have a directory on your machine where you store wheels,
called a `wheelhouse`.   The typical call for `wheel-uploader` would then
be something like:

```
wheel-uploader -v -w ~/wheelhouse netCDF4 1.1.8
```

where:

* `-v` means give verbose messages;
* `-w ~/wheelhouse` means download the wheels from https://wheels.scipy.org to
  the directory `~/wheelhouse`;
* `netCDF4` is the root name of the wheel(s) to download / upload;
* `1.1.8` is the version to download / upload.

So, in this case, `wheel-uploader` will download all wheels starting with
`netCDF4-1.1.8-` from http://wheels.scipy.org to `~/wheelhouse`, then upload
them to pypi.

Of course, you will need permissions to upload to pypi, for this to work.

[twine]: https://pypi.python.org/pypi/twine
[bs4]: https://pypi.python.org/pypi/beautifulsoup4

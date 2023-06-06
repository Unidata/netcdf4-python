# use pdoc (https://pdoc3.github.io/pdoc/) to generate API docs
pdoc3 --html --config show_source_code=False --force -o 'docs' netCDF4
/bin/cp -f docs/netCDF4/index.html docs/index.html

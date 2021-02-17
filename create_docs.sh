# Uses pdoc (https://github.com/mitmproxy/pdoc)
# to create html docs from docstrings in Cython source.
pdoc -o 'docs' netCDF4 
# use resulting docs/netCDF4/_netCDF4.html
cp docs/netCDF4.html docs/index.html
sed -i -e 's!href="../netCDF4.html!href="./index.html!g' docs/index.html
sed -i -e 's!/../netCDF4.html!/index.html!g' docs/index.html
sed -i -e 's!._netCDF4 API! API!g' docs/index.html
sed -i -e 's!netCDF4</a>._netCDF4</h1>!netCDF4</a></h1>!g' docs/index.html


#!/usr/bin/env python

import argparse
import netCDF4
import numpy
import os
import pickle
import scipy.sparse
import sys
import time

import regrid_tools as tools
from pykdtree.kdtree import KDTree

def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  """

  # Arguments
  parser = argparse.ArgumentParser(description=
      """
      regrid_runoff.py regrids runoff data from a regular/uniform latitude-longitude grid to a curvilinear ocean grid
      """,
      epilog='Written by A.Adcroft, 2018.')
  parser.add_argument('hgrid_file', type=str,
      help="""Filename for ocean horizontal grid (super-grid format).""")
  parser.add_argument('mask_file', type=str,
      help="""Filename for ocean mask.""")
  parser.add_argument('runoff_file', type=str,
      help="""Filename for gridded source runoff data.""")
  parser.add_argument('out_file', type=str,
      help="""Filename for runoff data on ocean model grid.""")
  parser.add_argument('-m','--mask_var', type=str, default='mask',
      help="""Name of mask variable in mask_file.""")
  parser.add_argument('-r','--runoff_var', type=str, default='runoff',
      help="""Name of runoff variable in runoff_file.""")
  parser.add_argument('-a','--ignore_area', action='store_true',
      help="""Ignore the "area" variable in the source runoff file.""")
  parser.add_argument('-f','--fast_pickle', action='store_true',
      help="""Use a pickled form of sparse matrix if available. This skips the matrix generation step if being re-applied to data.""")
  parser.add_argument('-c','--skip_coast', action='store_true',
      help="""Disable routing to nearest coastal cell.""")
  parser.add_argument('-n','--num_records', type=int, default=-1,
      help="""Number of records to write (default is to write all).""")
  parser.add_argument('--fms', action='store_true',
      help="""Add non-CF attributes to allow FMS to read data!""")
  parser.add_argument('--fmst', action='store_true',
      help="""Add non-CF modulo attributes to time for FMS!""")
  parser.add_argument('-z','--compress', action='store_true',
      help="""Use compressed file format.""")
  parser.add_argument('-p','--progress', action='store_true',
      help="""Report progress.""")
  parser.add_argument('-q','--quiet', action='store_true',
      help="""Disable informational messages.""")

  return parser.parse_args()

def info(msg):
  """Prints an informational message with trailing ... and no newline"""
  print(msg + ' ...', end='')
  sys.stdout.flush()
  return time.time()

def end_info(tic):
  """Closes the informational line"""
  print(' done in %.3fs.'%(time.time() - tic) )

def main(args):
  """
  Does everything.
  """

  start_time = time.time()
  pickle_file = 'pickle.regrid_runoff_A'

  # Open ocean grid
  if args.progress: tic = info('Reading ocean grid')
  ocn_qlon = netCDF4.Dataset(args.hgrid_file).variables['x'][:][::2,::2]   # Mesh longitudes (cell corners)
  ocn_qlat = netCDF4.Dataset(args.hgrid_file).variables['y'][:][::2,::2] # Mesh latitudes (cell corners)
  ocn_lon = netCDF4.Dataset(args.hgrid_file).variables['x'][:][1::2,1::2]    # Cell-center longitudes (cell centers)
  ocn_lat = netCDF4.Dataset(args.hgrid_file).variables['y'][:][1::2,1::2]  # Cell-center latitudes (cell centers)
  ocn_area = netCDF4.Dataset(args.hgrid_file).variables['area'][:]      # Super-grid cell areas
  ocn_area = ( ocn_area[::2,::2] + ocn_area[1::2,1::2] ) + ( ocn_area[1::2,::2] + ocn_area[::2,1::2] ) # Ocean-grid cell areas
  ocn_mask = netCDF4.Dataset(args.mask_file).variables[args.mask_var][:].astype(int) # 1=ocean, 0=land
  ocn_nj, ocn_ni = ocn_mask.shape
  ocn_id = numpy.arange( ocn_nj*ocn_ni ).reshape(ocn_mask.shape)
  ocn_qlon_bnds = ocn_qlon.min(), ocn_qlon.max()
  if args.progress: end_info(tic)

  if not args.quiet: print('Ocean grid shape is %i x %i.'%(ocn_nj, ocn_ni))

  # Read river grid
  if args.progress: tic = info('Reading runoff grid')
  runoff_file = netCDF4.Dataset(args.runoff_file)
  rvr_nj, rvr_ni = runoff_file.variables[args.runoff_var].shape[-2:]
  rvr_res = 360.0/rvr_ni

  # make rvr_lon  fit the same bounds as ocn
  rvr_lon = numpy.arange(0.5*rvr_res, 360., rvr_res)
  rvr_lon = numpy.where(rvr_lon < ocn_qlon_bnds[0], rvr_lon + 360, rvr_lon)
  rvr_lon = numpy.where(rvr_lon > ocn_qlon_bnds[1], rvr_lon - 360, rvr_lon)
  rvr_lat = numpy.arange(-90.+0.5*rvr_res, 90., rvr_res)

  # meshgrid as a list of 2d points (for constructing kdtree
  rvr_mesh = numpy.dstack(numpy.meshgrid(rvr_lat, rvr_lon)).reshape(-1, 2)

  rvr_qlon = numpy.arange(0, 360.0001, rvr_res)
  rvr_qlat = numpy.arange(-90., 90.001, rvr_res)

  rvr_nj, rvr_ni = rvr_lat.size, rvr_lon.size
  rvr_id = numpy.arange(rvr_nj*rvr_ni,dtype=numpy.int32) # River grid cell id
  if 'area' in runoff_file.variables:
    rvr_area = runoff_file.variables['area'][:]
  else:
    Re, d2r = 6371.26e3, numpy.pi/180  # File had area => Re = 6371.26043437 km
    rvr_area = numpy.outer( Re * ( numpy.sin( rvr_qlat[1:]*d2r ) - numpy.sin( rvr_qlat[:-1]*d2r ) ), Re*2*numpy.pi/rvr_ni*numpy.ones(rvr_ni))
    del d2r, Re
  if args.progress: end_info(tic)

  if not args.quiet: print('Runoff grid shape is %i x %i.'%(rvr_nj, rvr_ni))

  # Read cached regridding matrix
  do_full_calculation = True
  if args.fast_pickle and os.path.exists(pickle_file):
    if args.progress: tic = info('Reading pickled sparse matrix')
    f = open(pickle_file, 'rb')
    A = pickle.load( f )
    f.close()
    if args.progress: end_info(tic)
    if A.shape == (ocn_nj*ocn_ni,rvr_nj*rvr_ni):
      do_full_calculation = False
      if not args.quiet: print('Using cached regridding matrix.')

  if do_full_calculation:

    # Calculate river indices corresponding to each ocean cell
    if args.progress: tic = info('Calculating river cell id for each ocean cell')
    ocn_ri = ((1./rvr_res)*numpy.mod( ocn_lon, 360)).astype(int) # i-index in river-space of ocean cell
    ocn_rj = ((1./rvr_res)*( ocn_lat+90.)).astype(int) # j-index in river-space of ocean cell
    ocn_rid = ocn_ri + ocn_rj*rvr_ni
    del ocn_ri, ocn_rj
    if args.progress: end_info(tic)

    # Count ocean cell centers within each river cell
    if args.progress: tic = info('Calculating number of ocean cell centers within each river cell')
    rvr_ocells_in_rcells = numpy.zeros((rvr_nj*rvr_ni)) # Counter on river grid
    rids, cnt = numpy.unique( ocn_rid, return_counts=True )
    rvr_ocells_in_rcells[rids] = cnt
    del rids, cnt
    if args.progress: end_info(tic)

    if not args.quiet:
      cnt, freq = numpy.unique( rvr_ocells_in_rcells, return_counts=True )
      print('Frequency x river cells with # ocean cell centers:')
      for c,f in zip(cnt, freq):
        print('%ix%i'%(f,int(c)), end=' ')
      print()
      del cnt, freq

    # Calculate a coastal mask
    if args.progress: tic = info('Calculating a coastal mask')
    cst_mask = tools.coast_mask(ocn_mask)
    if args.progress: end_info(tic)

    if not args.quiet:
      print('There are %i/%i (%.2f%%) coastal cells on this ocean grid.'%(cst_mask.sum(),ocn_id.size,100*cst_mask.sum()/ocn_id.size))

    # Calculate nearest coastal cell
    if not args.skip_coast:
      if args.progress: tic = info('Calculating nearest coastal cell on ocean grid (should take ~%.1fs)'%(6*ocn_mask.size/(1080*1440)))
      cst_nrst_ocn_id = tools.nearest_coastal_cell( ocn_id, cst_mask )
      if args.progress: end_info(tic)

    # Build k-d tree for ocean grid
    if args.progress: tic = info('Building k-d tree for ocean grid (should take ~%.1fs)'%(210*ocn_mask.size/(1080*1440)))
    ocn_kdtree = KDTree(numpy.stack((ocn_qlat.flatten(), ocn_qlon.flatten()), axis=1))
    if args.progress: end_info(tic)

    # Find ocean cells corresponding to river cells
    if args.progress: tic = info('Using k-d tree to find ocean cells for each river cell (should take ~%.1fs)'%(40*ocn_mask.size/(1080*1440)))
    rvr_oid = ocn_kdtree.query(rvr_mesh)[1].reshape(rvr_nj, rvr_ni)
    if args.progress: end_info(tic)

    if not args.quiet:
      print('%i/%i river cells without associated ocean id.'%((rvr_oid<0).sum(),rvr_oid.size))

    # Construct sparse matrices
    if args.progress: tic = info('Constructing regridding matrix for river cells with many ocean cells')
    Arow = scipy.sparse.lil_matrix( (ocn_nj*ocn_ni, rvr_nj*rvr_ni), dtype=numpy.double )
    rids = rvr_id[rvr_ocells_in_rcells<=1].astype(numpy.int64)
    oids = rvr_oid.flatten()[rids].astype(numpy.int64)
    Arow[oids,rids] = rvr_area.flatten()[rids]
    del rids, oids
    if args.progress: end_info(tic)

    if args.progress: tic = info('Constructing regridding matrix for ocean cells with many river cells')
    Acol = scipy.sparse.lil_matrix( (ocn_nj*ocn_ni, rvr_nj*rvr_ni), dtype=numpy.double )
    rids = ocn_rid.flatten()
    oids = ocn_id.flatten() # debug without coastal mapping
    cnt = rvr_ocells_in_rcells.flatten()[rids]
    rids = rids[cnt>1]
    oids = oids[cnt>1]
    Acol[oids, rids] = rvr_area.flatten()[rids] / rvr_ocells_in_rcells[rids]
    del rids, oids, cnt
    if args.progress: end_info(tic)

    if not args.skip_coast:
      if args.progress: tic = info('Constructing rerouting matrix for coastal cells')
      Acst = scipy.sparse.lil_matrix( (ocn_nj*ocn_ni, ocn_nj*ocn_ni), dtype=numpy.double )
      Acst[cst_nrst_ocn_id, ocn_id] = 1
      if args.progress: end_info(tic)

    if args.progress: tic = info('Constructing final sparse matrix')
    if args.skip_coast:
      A = ( Arow + Acol )
    else:
      A = Acst * ( Arow + Acol )
      del Acst
    A = A.tocsr()
    del Acol, Arow
    if args.progress: end_info(tic)

    if args.progress: tic = info('Pickling sparse matrix')
    f = open(pickle_file, 'wb')
    pickle.dump( A, f )
    f.close()
    if args.progress: end_info(tic)

    if args.progress: print('Total pre-processing took %.1fs'%(time.time() - start_time))

  # Process runoff data
  if args.progress: tic = info('Regridding runoff and writing new file')
  totals = regrid_runoff(runoff_file, args.runoff_var, A, args.out_file, ocn_area, ocn_mask, ocn_qlat, ocn_qlon, ocn_lat, ocn_lon, rvr_area, args.fms, args.fmst, args.num_records, compress=args.compress )
  if args.progress: end_info(tic)

  if not args.quiet:
    for n in range(totals.shape[0]):
      err = abs( totals[n,0] - totals[n,1] ) / totals[n,0]
      print('Record %i, net source = %f kg/s, net regridded = %f kg/s, fractional error = %g'%(n,totals[n,0],totals[n,1],err))

def regrid_runoff( old_file, var_name, A, new_file_name, ocn_area, ocn_mask, ocn_qlat, ocn_qlon, ocn_lat, ocn_lon, rvr_area, fms_attr, fmst_attr, num_records , toler=1e-15, compress=False):
  """Regrids runoff data using sparse matrix A and write new file"""

  if compress:
    fileformat, zlib, complevel, area_dtype = 'NETCDF4', True, 1, 'f4'
  else:
    fileformat, zlib, complevel, area_dtype = 'NETCDF3_64BIT_OFFSET', None, None, 'd'
  new_file = netCDF4.Dataset(new_file_name, 'w', 'clobber', format=fileformat)
  runoff = old_file.variables[var_name]
  if len(runoff.shape) == 3:
    time = runoff.dimensions[0]
  else: time = None

  # Calculate shift to align left of source domain with Greenwich meridian
  if runoff.dimensions[-1] in old_file.variables:
    # Follows CF convention of 1d coordinate variable
    old_lon = old_file.variables[runoff.dimensions[-1]]
  else:
    # Work around for non CF-compliant file
    for lvar in ['lon','Lon','LON','longitude','Longitude','LONGITUDE','xc','XC','x','X']:
      if lvar in old_file.variables:
        break
    print(); print('WARNING! No coordinate variable for i-dimension. Using work around for "%s".'%(lvar)); print()
    if len(old_file.variables[lvar].shape) == 1:
      old_lon = old_file.variables[lvar][:]
    elif len(old_file.variables[lvar].shape) == 2:
      old_lon = old_file.variables[lvar][0,:]
    else:
      raise Exception('Coordinate variable has wrong shape!')
  dlon = 360. / old_lon.shape[0]
  ishift = numpy.argmin( numpy.abs( old_lon[:] - dlon*0.5) )
  del dlon, old_lon

  # Copy global attributes
  for a in old_file.ncattrs():
    new_file.setncattr(a, old_file.getncattr(a))

  # More info
  new_file.setncattr('regrid_note', 'Regridded with regrid_runoff.py')

  # Axes
  ocn_nj, ocn_ni = ocn_area.shape
  new_file.createDimension('i',ocn_ni)
  new_file.createDimension('j',ocn_nj)
  new_file.createDimension('IQ',ocn_ni+1)
  new_file.createDimension('JQ',ocn_nj+1)
  if time is not None:
    new_file.createDimension('time',None)

  # 1d variables
  i = new_file.createVariable('i', 'f4', ('i',))
  i.long_name = 'Grid position along first dimension'
  if fms_attr: i.cartesian_axis = 'X'
  j = new_file.createVariable('j', 'f4', ('j',))
  j.long_name = 'Grid position along second dimension'
  if fms_attr: j.cartesian_axis = 'Y'
  I = new_file.createVariable('IQ', 'f4', ('IQ',))
  I.long_name = 'Grid position along first dimension'
  J = new_file.createVariable('JQ', 'f4', ('JQ',))
  J.long_name = 'Grid position along second dimension'
  if time is not None:
    if '_FillValue' in old_file.variables[time].ncattrs():
      t = new_file.createVariable('time', 'd', ('time',), fill_value = old_file.variables[time]._FillValue)
    else:
      t = new_file.createVariable('time', 'd', ('time',))
    for a in old_file.variables[time].ncattrs():
      if a != '_FillValue':
        t.setncattr(a, old_file.variables[time].getncattr(a))
    t.long_name = 'Time'
    if fms_attr:
      t.cartesian_axis = 'T'
    if fmst_attr:
      t.modulo = ' '
      if num_records>0:
        t.modulo_beg = '1948-01-01 00:00:00'
        t.modulo_end = '%4.4i-01-01 00:00:00'%(1948+num_records/12)

  # 2d variables
  lon = new_file.createVariable('lon', 'f4', ('j','i',))
  lon.long_name = 'Longitude of cell centers'
  lon.standard_name = 'longitude'
  lon.units = 'degrees_east'
  lat = new_file.createVariable('lat', 'f4', ('j','i',))
  lat.long_name = 'Latitude of cell centers'
  lat.standard_name = 'latitude'
  lat.units = 'degrees_north'
  lonq = new_file.createVariable('lon_crnr', 'f4', ('JQ','IQ',))
  lonq.long_name = 'Longitude of mesh nodes'
  lonq.standard_name = 'longitude'
  lonq.units = 'degrees_east'
  latq = new_file.createVariable('lat_crnr', 'f4', ('JQ','IQ',))
  latq.long_name = 'Latitude of mesh nodes'
  latq.standard_name = 'latitude'
  latq.units = 'degrees_north'
  area = new_file.createVariable('area', area_dtype, ('j','i',))
  area.long_name = 'Cell area'
  area.standard_name = 'cell_area'
  area.units = 'm2'
  area.coordinates = 'lon lat'
  area.mesh_coordinates = 'lon_crnr lat_crnr'
  if time is not None:
    dims = ('time', 'j','i',)
  else:
    dims = ('j','i',)
  if '_FillValue' in old_file.variables[var_name].ncattrs():
    new_runoff = new_file.createVariable(var_name, 'f4', dims, fill_value = old_file.variables[var_name]._FillValue, zlib=zlib, complevel=complevel)
    if fms_attr: new_runoff.missing_value = old_file.variables[var_name]._FillValue
  else:
    new_runoff = new_file.createVariable(var_name, 'f4', dims, zlib=zlib, complevel=complevel)
  new_runoff.coordinates = 'lon lat'
  new_runoff.mesh_coordinates = 'lon_crnr lat_crnr'
  new_runoff.standard_name = 'runoff_flux'

  # runoff attributes
  for a in old_file.variables[var_name].ncattrs():
    if a != '_FillValue':
      new_runoff.setncattr(a, old_file.variables[var_name].getncattr(a))
  if runoff.units == 'kg/s/m^2': new_runoff.units = 'kg m-2 s-1'

  # Write static data
  i[:] = numpy.arange(ocn_ni)+0.5
  j[:] = numpy.arange(ocn_nj)+0.5
  I[:] = numpy.arange(ocn_ni+1)
  J[:] = numpy.arange(ocn_nj+1)
  lon[:,:] = ocn_lon
  lat[:,:] = ocn_lat
  lonq[:,:] = ocn_qlon
  latq[:,:] = ocn_qlat
  area[:,:] = ocn_area

  i_area = 1/ocn_area

  nrecs = runoff.shape[0]
  if num_records>0: nrecs = num_records
  totals = numpy.zeros((nrecs,2))
  for n in range(nrecs):
    if ishift == 0:
      data = runoff[n]
    else:
      data = numpy.roll( runoff[n], -ishift, axis=-1 )
    tim = old_file.variables[time][n]
    odata = ( A * data.flatten() ).reshape(ocn_area.shape)
    odata = numpy.ma.array( odata, mask=(ocn_mask==0) )
    new_runoff[n] = i_area * odata
    t[n] = tim
    totals[n,0] = (rvr_area * data ).sum()
    totals[n,1] = odata.sum()
    err = abs( totals[n,0] - totals[n,1] ) / totals[n,0]
    if err>toler:
      print('Non-conservation for record %i, net source = %f kg/s, net regridded = %f kg/s, fractional error = %g'%(n,totals[n,0],totals[n,1],err))

  new_file.close()

  return totals

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__':
  args = parseCommandLine()
  main(args)

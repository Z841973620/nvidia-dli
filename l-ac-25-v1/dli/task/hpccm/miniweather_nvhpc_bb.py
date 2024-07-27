# Development stage
Stage0 += baseimage(image='ubuntu:18.04', _as='build')

# NVDIDIA HPC SDK
Stage0 += nvhpc(eula=True, redist=['compilers/lib/*'], version='20.7')

# Parallel NetCDF
Stage0 += pnetcdf()

# CMake
Stage0 += cmake(eula=True)

# MiniWeather
Stage0 += packages(ospackages=['ca-certificates', 'git'])

# MiniWeather
Stage0 += generic_cmake(cmake_opts=['-DCMAKE_Fortran_COMPILER=mpif90',
                                    '-DDATA_SPEC=DATA_SPEC_INJECTION',
                                    '-DPNETCDF_PATH=/usr/local/pnetcdf',
                                    '-DOPENACC_FLAGS="-acc -gpu=ccall,ptxinfo"'],
                        commit='b4dcb664e559bce0b1aa8d8b5d466d2303dd43cf',
                        directory='miniWeather/fortran',
                        prefix='/opt/miniWeather',
                        repository='https://github.com/mrnorman/miniWeather',
                        # workaround lack of install target
                        install=False,
                        postinstall=['install -m 755 -d /opt/miniWeather/bin',
                                     'install -m 755 /var/tmp/miniWeather/fortran/build/openacc /opt/miniWeather/bin/openacc'],
                        preconfigure=['mkdir -p /opt/miniWeather'])

# Runtime stage
Stage1 += baseimage(image='ubuntu:18.04')

# Runtime components from the previous stage
Stage1 += Stage0.runtime(_from='build')

# MiniWeather
Stage1 += environment(variables={'PATH': '/opt/miniWeather/bin:$PATH'})

# nvidia-container-runtime
Stage1 += environment(variables={
  'NVIDIA_VISIBLE_DEVICES': 'all',
  'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility',
  'NVIDIA_REQUIRE_CUDA': '"cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"'})



cython <cython_file> --embed
gcc <C_file_from_cython> -I<include_directory> -L<directory_containing_libpython> -l<name_of_libpython_without_lib_on_the_front> -o <output_file_name>

#numpy header file dependencies are returned by numpy.get_include()

gcc <C_file_from_cython> -I<include_directory> -I<numpy_include_directory> -L<directory_containing_libpython> -l<name_of_libpython_without_lib_on_the_front> -o <output_file_name>

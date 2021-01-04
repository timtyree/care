python3 setup.py build_ext --inplace 
# gcc -p thread -Wno-unused-result -Wsign-compare -DDYNAMIC_ANNOTATIONS_ENABLED=1 -DNDEBUG -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=generic -D_GNU_SOURCE -fPIC -fwrapv -fPIC -I/usr/include/python3.6m -c my_module.c --inplace
# -o build/temp.linux-x86_64-3.6/my_module.o

#-lboost_python38 -lpython3.8
# python3 setup.py build_ext -lboost_python38 --inplace 
